import os

from isaaclab.sim.spawners.from_files import from_files as _from_files

# Bundled copy of the Isaac Sim grid ground plane, downloaded from
# {ISAAC_NUCLEUS_DIR}/Environments/Grid so flat-terrain scenes do not depend on
# the Omniverse asset server being reachable.
_LOCAL_GROUND_PLANE_USD = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "assets", "ground_plane", "Grid", "default_environment.usd")
)

_REMOTE_GROUND_PLANE_SUFFIX = "Environments/Grid/default_environment.usd"


def patch_ground_plane_to_local_asset():
    """Spawn the grid ground plane from the bundled copy instead of the asset server.

    The default GroundPlaneCfg points at a USD hosted on NVIDIA's remote asset
    server. When that URL cannot be reached (blocked IPv6, firewall, offline
    machine), scene creation fails with:
        FileNotFoundError: Unable to open the usd file at path: https://...
    The spawn function is resolved lazily from the from_files module, so
    wrapping it here redirects every terrain_type="plane" scene to the local
    asset. Does nothing if the bundled files are missing.
    """
    if not os.path.isfile(_LOCAL_GROUND_PLANE_USD):
        return

    original_spawn_ground_plane = _from_files.spawn_ground_plane

    def spawn_ground_plane(prim_path, cfg, *args, **kwargs):
        if cfg.usd_path and cfg.usd_path.replace(os.sep, "/").endswith(_REMOTE_GROUND_PLANE_SUFFIX):
            cfg.usd_path = _LOCAL_GROUND_PLANE_USD
        return original_spawn_ground_plane(prim_path, cfg, *args, **kwargs)

    _from_files.spawn_ground_plane = spawn_ground_plane
