# Source Layout

The importable package under `isaac_underwater/` is split by ownership:

`robot`, `physics`, `actuators`, `sensors`, `appearance`, `worlds`,
`localization`, `controllers`, `navigation`, `randomization`, `interfaces`,
and `tasks`. Keeping these as package modules lets Isaac Lab tasks and future
real-robot adapters share the same contracts without a monolithic launcher.
`localization` includes the GT backend, namespace-aware external VIO transport,
and a deliberately degraded IMU dead-reckoning adapter for IPC tests; the
adapter is not presented as a VIO algorithm.
