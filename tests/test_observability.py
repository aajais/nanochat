"""CPU-only contract tests; no W&B writes or paid sandbox creation.

python -m unittest discover -s tests -p 'test_observability.py' -v
"""

import copy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import wandb
from nanochat import observability as obs
from nanochat.calculator import safe_calculator
from nanochat.rollouts import ASSETS, ROOT, RolloutConfig, RolloutPool, sandbox_options, validate_response
from scripts.rollout_worker import generate_jobs


class Result:
    def __init__(self, value=None):
        self.value = value

    def result(self):
        return self.value


class FakeSandbox:
    instances = []
    snapshots = {}
    options = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop().result()

    @classmethod
    def run(cls, **options):
        cls.options.append(options)
        worker = cls()
        worker.files = {}
        if "file_system_snapshot" in options:
            source = options["file_system_snapshot"].file_system_snapshot_id
            worker.files = copy.deepcopy(cls.snapshots.get(source, {}))
        worker.sandbox_id = f"sandbox-{len(cls.instances)}"
        worker.stopped = False
        cls.instances.append(worker)
        return worker

    @classmethod
    def get_snapshot(cls, snapshot_id, **kwargs):
        assert snapshot_id in cls.snapshots
        return Result(SimpleNamespace(status="READY"))

    def wait(self, **kwargs):
        return None

    def exec(self, argv, **kwargs):
        if "--submit" in argv:
            request = json.loads(self.files[f"{ROOT}/request.tmp"])
            response = {"request_id": request["request_id"], "version": request["version"],
                        "results": [{"job_id": j["job_id"],
                                     "sequences": [j["tokens"] + [3] for _ in range(j["num_samples"])],
                                     "masks": [[0] * len(j["tokens"]) + [1] for _ in range(j["num_samples"]) ]}
                                    for j in request["jobs"]]}
            self.files[f"{ROOT}/response.json"] = json.dumps(response).encode()
        return Result(SimpleNamespace(returncode=0, stdout="", stderr=""))

    def write_file(self, path, content):
        self.files[path] = content
        return Result()

    def write_file_streaming(self, path, chunks, **kwargs):
        self.files[path] = b"".join(chunks)
        return Result()

    def read_file(self, path):
        return Result(self.files[path])

    def snapshot(self, **kwargs):
        assert kwargs == {"wait_for_ready": True}
        snapshot_id = f"fss-{len(self.snapshots)}"
        self.snapshots[snapshot_id] = {p: data for p, data in self.files.items() if p.startswith(ASSETS + "/")}
        return Result(snapshot_id)

    def stop(self, **kwargs):
        self.stopped = True
        return Result()


def fake_sdk():
    return SimpleNamespace(Sandbox=FakeSandbox, ResourceOptions=lambda **kw: SimpleNamespace(**kw),
        NetworkOptions=lambda **kw: SimpleNamespace(**kw), FileSystemSnapshotOptions=lambda **kw: SimpleNamespace(**kw),
        AuthStrategy=SimpleNamespace(COREWEAVE_API_KEY="coreweave", WANDB="wandb"),
        SandboxTimeoutError=TimeoutError)


def job(index):
    return {"job_id": str(index), "tokens": [1, 2], "num_samples": 2, "batch_size": 1,
            "max_tokens": 4, "temperature": 1.0, "top_k": 50, "seeds": [123, 124]}


@dataclass
class TinyConfig:
    vocab_size: int = 8


class RolloutTests(unittest.TestCase):
    def setUp(self):
        FakeSandbox.instances, FakeSandbox.snapshots, FakeSandbox.options = [], {}, []
        self.sdk = patch.dict("sys.modules", {"cwsandbox": fake_sdk()})
        self.sdk.start()
        self.auth = patch.dict(os.environ, {"CWSANDBOX_API_KEY": "test-not-a-credential"})
        self.auth.start()
        self.run = patch.object(obs, "active_run", return_value=None)
        self.run.start()
        self.temp = tempfile.TemporaryDirectory()
        self.tokenizer = Path(self.temp.name) / "tokenizer.pkl"
        self.tokenizer.write_bytes(b"test-tokenizer")
        self.base = patch("nanochat.common.get_base_dir", return_value=self.temp.name)
        self.base.start()
        self.model = torch.nn.Linear(2, 2)
        self.model.config = TinyConfig()

    def tearDown(self):
        self.base.stop()
        self.temp.cleanup()
        self.run.stop()
        self.auth.stop()
        self.sdk.stop()

    def config(self, **kwargs):
        return RolloutConfig(image="registry/worker@sha256:" + "a" * 64, **kwargs)

    def test_options_no_credentials_and_denied_network(self):
        options = sandbox_options(self.config())
        self.assertEqual(options["placement_spillover"], "cks_then_serverless")
        self.assertEqual(options["resources"].gpu, {"count": 1})
        self.assertTrue(options["network"].deny_egress)
        self.assertTrue(options["network"].deny_ingress)
        self.assertNotIn("CWSANDBOX_API_KEY", options["environment_variables"])
        self.assertNotIn("WANDB_API_KEY", options["environment_variables"])
        self.assertNotIn("file_system_snapshot", options)

    def test_runner_pinning_requires_strict_cks(self):
        with self.assertRaises(ValueError):
            self.config(runner_ids=("runner-1",))
        options = sandbox_options(self.config(placement="cks", runner_ids=("runner-1",)))
        self.assertEqual(options["runner_ids"], ["runner-1"])
        self.assertEqual(options["placement_spillover"], "strict")

    def test_reject_mutable_image_and_unbounded_workers(self):
        with self.assertRaises(ValueError):
            RolloutConfig(image="registry/worker:latest")
        with self.assertRaises(ValueError):
            self.config(workers=33)

    def test_current_policy_order_and_cleanup(self):
        pool = RolloutPool(self.config(workers=2), self.tokenizer)
        try:
            with self.assertRaises(RuntimeError):
                pool.generate_many([job(0)])
            pool.sync(self.model, 0)
            first_version = pool.version
            results = pool.generate_many([job(i) for i in range(5)])
            self.assertEqual([r["job_id"] for r in results], [str(i) for i in range(5)])
            self.assertEqual(len(FakeSandbox.instances), 2)
            pool.sync(self.model, 1)
            self.assertNotEqual(pool.version, first_version)
            state = FakeSandbox.instances[0].files[f"{ROOT}/policy.pt"]
            self.assertGreater(len(state), 0)
        finally:
            pool.close()
        self.assertTrue(all(w.stopped for w in FakeSandbox.instances))

    def test_snapshot_round_trip_and_fresh_weights(self):
        pool = RolloutPool(self.config(save_snapshot=True), self.tokenizer)
        pool.sync(self.model, 0)
        pool.generate_many([job(0)])
        snapshot_id = pool.snapshot_id
        self.assertIsNotNone(snapshot_id)
        self.assertNotIn(f"{ROOT}/policy.pt", FakeSandbox.snapshots[snapshot_id])
        self.assertNotIn(f"{ROOT}/response.json", FakeSandbox.snapshots[snapshot_id])
        pool.close()
        restored = RolloutPool(self.config(snapshot_id=snapshot_id, workers=2), self.tokenizer)
        try:
            with self.assertRaises(RuntimeError):
                restored.generate_many([job(0)])
            restored.sync(self.model, 42)
            self.assertEqual(len(restored.generate_many([job(0)])), 1)
        finally:
            restored.close()

    def test_snapshot_mismatch_stops_partial_startup(self):
        FakeSandbox.snapshots["fss-bad"] = {f"{ASSETS}/manifest.json": b"{}"}
        with self.assertRaises(ValueError):
            RolloutPool(self.config(snapshot_id="fss-bad"), self.tokenizer)
        self.assertTrue(all(w.stopped for w in FakeSandbox.instances))

    def test_generation_error_stops_pool_without_fallback(self):
        pool = RolloutPool(self.config(), self.tokenizer)
        pool.sync(self.model, 0)
        with patch.object(pool, "_request", side_effect=TimeoutError("timeout")):
            with self.assertRaises(TimeoutError):
                pool.generate_many([job(0)])
        self.assertTrue(all(w.stopped for w in FakeSandbox.instances))

    def test_reject_stale_response_and_bad_masks(self):
        request = {"request_id": "current", "version": "v1", "model_config": {"vocab_size": 8}, "jobs": [job(0)]}
        response = {"request_id": "old", "version": "v1", "results": []}
        with self.assertRaises(RuntimeError):
            validate_response(response, request)
        response.update(request_id="current", results=[{"job_id": "0", "sequences": [[1, 2, 3]] * 2,
                                                        "masks": [[1, 0, 1]] * 2}])
        with self.assertRaises(RuntimeError):
            validate_response(response, request)
        response["results"][0]["masks"] = [[0, 0, 1]] * 2
        self.assertEqual(len(validate_response(response, request)), 1)

    def test_worker_preserves_microbatches_and_seeds(self):
        calls = []
        def generate(tokens, **kwargs):
            calls.append(kwargs)
            return [tokens + [3]], [[0] * len(tokens) + [1]]
        results = generate_jobs(SimpleNamespace(generate_batch=generate), [job(0)])
        self.assertEqual([c["seed"] for c in calls], [123, 124])
        self.assertEqual(len(results[0]["sequences"]), 2)

    def test_generated_code_forces_serverless_and_cleans_up(self):
        from nanochat.execution import execute_code
        with patch.dict(os.environ, {"NANOCHAT_EXECUTION_BACKEND": "cwsandbox"}), patch.object(FakeSandbox, "exec", return_value=Result(SimpleNamespace(returncode=0, stdout="4\n", stderr=""))):
            result = execute_code("print(2 + 2)")
        self.assertTrue(result.success)
        self.assertEqual(result.stdout, "4\n")
        self.assertEqual(FakeSandbox.options[0]["placement_mode"], "serverless")
        self.assertTrue(FakeSandbox.instances[0].stopped)

    def test_generated_code_timeout_is_preserved(self):
        from nanochat.execution import execute_code
        with patch.dict(os.environ, {"NANOCHAT_EXECUTION_BACKEND": "cwsandbox"}), patch.object(FakeSandbox, "exec", side_effect=TimeoutError("timeout")):
            result = execute_code("print(2 + 2)")
        self.assertTrue(result.timeout)
        self.assertFalse(result.success)
        self.assertTrue(FakeSandbox.instances[0].stopped)

    def test_service_failure_never_runs_generated_code_locally(self):
        from nanochat.execution import execute_code
        with patch.dict(os.environ, {"NANOCHAT_EXECUTION_BACKEND": "cwsandbox"}), patch.object(FakeSandbox, "run", side_effect=RuntimeError("service unavailable")), patch("subprocess.run") as local:
            with self.assertRaisesRegex(RuntimeError, "service unavailable"):
                execute_code("print(2 + 2)")
            local.assert_not_called()


class ObservabilityTests(unittest.TestCase):
    def test_dummy_run_never_initializes_wandb(self):
        with patch.object(wandb, "init") as initialize:
            run = obs.init_run("nanochat", "dummy", {}, enabled=False)
            run.log({"step": 0})
            initialize.assert_not_called()

    def test_suppressed_example_does_not_trace_nested_execution(self):
        with patch.object(obs, "_client", SimpleNamespace(create_call=lambda *a, **kw: self.fail("unexpected call"))):
            with obs.trace_span("unsampled-score", {}, enabled=False):
                with obs.trace_span("execute", {}):
                    pass
        self.assertFalse(obs._suppressed.get())

    def test_core_correctness_unchanged_with_telemetry(self):
        from nanochat import core_eval
        item = {"query": "question", "choices": ["yes", "no"], "gold": 0}
        tokenizer = SimpleNamespace(get_bos_token_id=lambda: 0)
        task = {"task_type": "multiple_choice", "num_fewshot": 0, "continuation_delimiter": " "}
        logs = []
        with patch.object(core_eval, "batch_sequences_mc", return_value=([[1, 2], [1, 3]], [1, 1], [2, 2])), patch.object(core_eval, "forward_model", return_value=(torch.tensor([[0.1, 0.0], [2.0, 0.0]]), torch.zeros(2, 2))), patch.object(obs, "active_run", return_value=SimpleNamespace(log=logs.append)):
            with obs.evaluation("CORE/test"):
                outcome = core_eval.evaluate_example(0, object(), tokenizer, [item], "cpu", task)
        self.assertTrue(outcome)
        self.assertTrue(logs[0]["eval/CORE/test/examples"].data[0][4])

    def test_local_code_execution_and_invalid_backend(self):
        from nanochat.execution import execute_code
        with patch.dict(os.environ, {"NANOCHAT_EXECUTION_BACKEND": "local"}):
            result = execute_code("print(2 + 2)")
            self.assertTrue(result.success)
            self.assertEqual(result.stdout, "4\n")
        with patch.dict(os.environ, {"NANOCHAT_EXECUTION_BACKEND": "unknown"}):
            with self.assertRaises(ValueError):
                execute_code("print(2 + 2)")

    def test_eval_rows_bounded_and_private_by_default(self):
        run = SimpleNamespace(log=lambda data: None)
        with patch.object(obs, "active_run", return_value=run), patch.dict(os.environ, {"NANOCHAT_EVAL_ROWS": "1", "NANOCHAT_LOG_CONTENT": "0"}):
            report = obs.Evaluation("task", 10, {})
            for i in range(3):
                report.record_row(i, 0, "private-prompt", "reference", "answer", True, 2, 3, 0.1)
            self.assertEqual(len(report.rows), 1)
            self.assertEqual(report.rows[0][1:4], [None, None, None])
            logs = []
            run.log = logs.append
            report.log(0.3)
            table = logs[0]["eval/task/examples"]
            self.assertEqual(table.data, report.rows)
            self.assertEqual(logs[0]["step"], 10)

    def test_content_opt_in_and_legacy_table_fallback(self):
        logs = []
        legacy = SimpleNamespace(Table=wandb.Table)
        with patch.object(obs, "active_run", return_value=SimpleNamespace(log=logs.append)), patch.object(obs, "wandb", legacy), patch.dict(os.environ, {"NANOCHAT_LOG_CONTENT": "1"}):
            report = obs.Evaluation("task", 0, {})
            report.record_row(0, 0, "prompt", "reference", "answer", False)
            with self.assertWarns(UserWarning):
                report.log(0.1)
            self.assertEqual(logs[0]["eval/task/examples"].data[0][1:4], ["prompt", "reference", "answer"])

    def test_rank_nonzero_has_no_active_run(self):
        with patch.dict(os.environ, {"RANK": "1"}), patch.object(wandb, "run", object()):
            self.assertIsNone(obs.active_run())

    def test_nested_trace_parentage_exception_and_context_reset(self):
        created, finished = [], []
        def create(name, **kwargs):
            call = SimpleNamespace(name=name, **kwargs)
            created.append(call)
            return call
        client = SimpleNamespace(create_call=create, finish_call=lambda call, **kwargs: finished.append((call, kwargs)))
        with patch.object(obs, "_client", client), patch.object(obs, "active_run", return_value=None):
            with self.assertRaisesRegex(ValueError, "expected"):
                with obs.trace_span("parent", {}):
                    with obs.trace_span("child", {}):
                        raise ValueError("expected")
            self.assertIs(created[1].parent, created[0])
            self.assertIsNone(obs._parent.get())
            self.assertEqual(len(finished), 2)
            self.assertTrue(all(isinstance(kwargs["exception"], ValueError) for _, kwargs in finished))

    def test_gradient_diagnostics_do_not_change_gradients(self):
        model = torch.nn.Linear(2, 1)
        model.get_device = lambda: torch.device("cpu")
        model(torch.ones(1, 2)).sum().backward()
        before = [p.grad.clone() for p in model.parameters()]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        with patch.object(obs, "active_run", return_value=object()):
            metrics = obs.optimizer_metrics(model, optimizer)
        self.assertAlmostEqual(metrics["train/grad_norm_rank0"], 3 ** 0.5, places=6)
        self.assertEqual(metrics["train/grad_nonfinite_rank0"], 0)
        self.assertTrue(all(torch.equal(p.grad, g) for p, g in zip(model.parameters(), before)))

    def test_calculator_interprets_only_bounded_arithmetic_and_count(self):
        self.assertEqual(safe_calculator("(1,000 + 2) / 2"), 501)
        self.assertEqual(safe_calculator("'banana'.count('a')"), 3)
        self.assertIsNone(safe_calculator("2 ** 1000000"))
        self.assertIsNone(safe_calculator("[x for x in range(10)]"))
        self.assertIsNone(safe_calculator("1 / 0"))


if __name__ == "__main__":
    unittest.main()