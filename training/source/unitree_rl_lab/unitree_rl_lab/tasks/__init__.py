##
# Register Gym environments.
##

from isaaclab_tasks.utils import import_packages

from unitree_rl_lab.utils.local_ground_plane import patch_ground_plane_to_local_asset

# Serve the grid ground plane from the bundled asset so flat-terrain scenes
# work without access to the Omniverse asset server.
patch_ground_plane_to_local_asset()

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = []
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
