"""CPU/Gloo regression for remote I/O exceeding the collective timeout."""
from datetime import timedelta
from contextlib import contextmanager
import json
import multiprocessing as mp
from pathlib import Path
import signal
import time

import pytest
import torch
import torch.distributed as dist

from nanochat import sft_runtime


def _worker(rank, root, mode):
    root = Path(root)
    dist.init_process_group(
        'gloo', init_method=(root / 'rendezvous').as_uri(), rank=rank,
        world_size=2, timeout=timedelta(seconds=1),
    )
    try:
        # Agree on a stage deadline before remote I/O begins.
        end = torch.tensor(time.time() + (2 if mode == 'timeout' else 10), dtype=torch.float64)
        dist.broadcast(end, 0)

        def publish():
            if mode == 'failure':
                raise ValueError('Server checkpoint bytes differ')
            if mode == 'timeout':
                # Models an SDK retry loop: deadline must escape Exception catches.
                while True:
                    try:
                        time.sleep(10)
                    except Exception:
                        continue
            time.sleep(2)  # Longer than the process-group timeout.
            return {'server_verified': True, 'ref': 'test/checkpoint:v0'}

        receipt = sft_runtime.coordinate_publication(root / 'publication.json', publish, end.item())
        # All ranks may resume collectives only after publication is verified.
        value = torch.tensor(rank + 1)
        dist.all_reduce(value)
        result = {'receipt': receipt, 'sum': value.item()}
    except BaseException as exc:
        result = {'error': type(exc).__name__, 'message': str(exc)}
    finally:
        (root / f'result-{rank}.json').write_text(json.dumps(result))
        dist.destroy_process_group()


def _run_ranks(tmp_path, mode):
    # A missing implementation is an explicit failing contract, not a collection error.
    assert callable(getattr(sft_runtime, 'coordinate_publication', None)), 'Missing deadline-bounded checkpoint coordination'
    ctx = mp.get_context('spawn')
    workers = [ctx.Process(target=_worker, args=(rank, str(tmp_path), mode)) for rank in range(2)]
    for worker in workers:
        worker.start()
    try:
        deadline = time.monotonic() + 30
        for worker in workers:
            worker.join(max(0, deadline - time.monotonic()))
        assert all(not worker.is_alive() for worker in workers), 'Checkpoint coordination hung'
        assert [worker.exitcode for worker in workers] == [0, 0]
        return [json.loads((tmp_path / f'result-{rank}.json').read_text()) for rank in range(2)]
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.kill()
            worker.join()


def test_slow_publication_does_not_hold_a_collective(tmp_path):
    results = _run_ranks(tmp_path, 'success')
    assert results == [{'receipt': {'server_verified': True, 'ref': 'test/checkpoint:v0'}, 'sum': 3}] * 2


def test_publication_failure_reaches_every_rank(tmp_path):
    results = _run_ranks(tmp_path, 'failure')
    assert all('Server checkpoint bytes differ' in r.get('message', '') for r in results)
    assert all('receipt' not in r for r in results)
    assert json.loads((tmp_path / 'publication.json').read_text())['passed'] is False


def test_stalled_publication_expires_on_all_ranks(tmp_path):
    results = _run_ranks(tmp_path, 'timeout')
    assert all(r.get('error') == 'TimeoutError' for r in results)
    assert all('receipt' not in r for r in results)


def test_expired_deadline_never_calls_publisher(tmp_path):
    assert callable(getattr(sft_runtime, 'coordinate_publication', None))
    called = []
    with pytest.raises(TimeoutError):
        sft_runtime.coordinate_publication(tmp_path / 'publication.json', lambda: called.append(True), time.time() - 1)
    assert not called


def test_alarm_handler_and_sigterm_handler_are_preserved(tmp_path):
    assert callable(getattr(sft_runtime, 'coordinate_publication', None))
    alarm = signal.getsignal(signal.SIGALRM)
    term = signal.getsignal(signal.SIGTERM)
    result = sft_runtime.coordinate_publication(tmp_path / 'publication.json', lambda: {'ok': True}, time.time() + 5)
    assert result == {'ok': True}
    assert signal.getsignal(signal.SIGALRM) == alarm
    assert signal.getsignal(signal.SIGTERM) == term
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_deadline_exit_failure_never_exposes_success(tmp_path, monkeypatch):
    @contextmanager
    def expires_on_exit(_):
        yield
        raise TimeoutError('Deadline reached after publication')

    observed = []
    write = sft_runtime.atomic_json

    def observe(path, value):
        write(path, value)
        observed.append(json.loads(path.read_text())['passed'])

    monkeypatch.setattr(sft_runtime, '_publication_deadline', expires_on_exit)
    monkeypatch.setattr(sft_runtime, 'atomic_json', observe)
    with pytest.raises(TimeoutError):
        sft_runtime.coordinate_publication(tmp_path / 'status.json', lambda: {}, time.time() + 5)
    assert observed == [False], 'Peers must never see a success that can be revoked'


def test_missing_publisher_expires_without_collective(tmp_path, monkeypatch):
    monkeypatch.setattr(sft_runtime, 'rank_world', lambda: (1, 2))
    with pytest.raises(TimeoutError):
        sft_runtime.coordinate_publication(tmp_path / 'missing.json', None, time.time() + .1)


def test_stale_success_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(sft_runtime, 'rank_world', lambda: (1, 2))
    path = tmp_path / 'status.json'
    path.write_text(json.dumps({'passed': True, 'deadline_unix': 1, 'receipt': {}}))
    with pytest.raises(ValueError, match='Stale'):
        sft_runtime.coordinate_publication(path, None, time.time() + 5)


def test_existing_alarm_is_not_extended(tmp_path):
    previous = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, lambda *_: None)
    signal.setitimer(signal.ITIMER_REAL, 5)
    try:
        with pytest.raises(RuntimeError, match='active alarm'):
            sft_runtime.coordinate_publication(tmp_path / 'status.json', lambda: {}, time.time() + 50)
        assert 0 < signal.getitimer(signal.ITIMER_REAL)[0] <= 5
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
