# Environment index

The authoritative migration and rebuild guide is [README.md](README.md).

| Environment | Python | Purpose | Rebuild input |
| --- | --- | --- | --- |
| `.venvs/piper-home` | 3.10.12 | Piper/Orbbec recording, viewer and hardware bridge | `environment-locks/piper-home.runtime-lock.txt` + editable `pyAgxArm/` |
| `.venvs/lerobot-pi05` | 3.12.14 | LeRobot data conversion, PI05 training and inference | `environment-locks/lerobot-pi05.runtime-lock.txt` + editable `lerobot/` and `pyAgxArm/` |

Create or verify them with:

```bash
bash thirdParty/setup_envs.sh
bash thirdParty/setup_envs.sh --check
```

Do not copy `.venvs` between machines. Recreate them from the recorded lock files.
