"""Exercise chat-eval CLI logging without model assets, GPUs or service writes.

python -m unittest discover -s tests -p 'test_eval_checkpoint_step.py' -v
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch  # Load extension modules before patch.dict restores sys.modules.
from nanochat import common
from nanochat import observability as obs
from tasks import arc, gsm8k, humaneval, mmlu


class CheckpointStepTests(unittest.TestCase):
    def run_eval(self, requested_step, loaded_step):
        # Mock only inference/checkpoint boundaries; execute the real main(),
        # evaluation wrapper, EvalTable construction and trace creation path.
        spec = importlib.util.spec_from_file_location(
            "chat_eval_under_test", Path(__file__).resolve().parents[1] / "scripts/chat_eval.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {
            "nanochat.checkpoint_manager": SimpleNamespace(load_model=Mock()),
            "nanochat.engine": SimpleNamespace(Engine=Mock()),
        }):
            spec.loader.exec_module(module)

        run = Mock(id="eval-test")
        client = Mock()
        argv = ["chat_eval", "--source", "sft", "--run", "step-test", "--device-type", "cpu"]
        if requested_step is not None:
            argv += ["--step", str(requested_step)]

        def inference(*args):
            obs._evaluation.get().record_row(0, 0, "prompt", "reference", "answer", True)
            return 1.0

        with patch("sys.argv", argv), \
             patch.object(module, "compute_init", return_value=(False, 0, 0, 1, "cpu")), \
             patch.object(module, "compute_cleanup"), \
             patch.object(module, "print0"), \
             patch.object(module, "init_run", return_value=run), \
             patch.object(module, "log_model_config") as config, \
             patch.object(module, "_run_chat_eval", side_effect=inference), \
             patch.object(obs, "active_run", return_value=run), \
             patch.object(obs, "_client", client), \
             patch("nanochat.artifact_tracking.log_evaluation") as artifact:
            module.load_model.return_value = (object(), object(), {"step": loaded_step})
            module.main()

        self.assertEqual(module.load_model.call_args.kwargs["step"], requested_step)
        self.assertEqual(config.call_args.kwargs["step"], loaded_step)
        # Keep the CLI request distinguishable from the resolved checkpoint.
        self.assertEqual(config.call_args.args[1].step, requested_step)
        logs = [call.args[0] for call in run.log.call_args_list]
        self.assertTrue(all(log["step"] == loaded_step for log in logs))
        self.assertEqual(sum("chatcore_metric" in log for log in logs), 1)
        tables = [value for log in logs for key, value in log.items() if key.endswith("/examples")]
        self.assertEqual(len(tables), 5)
        self.assertTrue(all(len(table.data) == 1 for table in tables))
        self.assertEqual(artifact.call_count, 5)
        self.assertTrue(all(call.args[2] == loaded_step for call in artifact.call_args_list))
        self.assertEqual(client.create_call.call_count, 5)
        self.assertTrue(all(call.kwargs["inputs"]["step"] == loaded_step
                            for call in client.create_call.call_args_list))
        run.finish.assert_called_once()

    def test_omitted_step_logs_loaded_checkpoint_in_all_outputs(self):
        self.run_eval(requested_step=None, loaded_step=2)

    def test_explicit_step_zero_is_not_replaced(self):
        self.run_eval(requested_step=0, loaded_step=0)

    def test_explicit_nonzero_step(self):
        self.run_eval(requested_step=2, loaded_step=2)


if __name__ == "__main__":
    unittest.main()