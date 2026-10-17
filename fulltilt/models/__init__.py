# ========================================================================
# The copyrights of the software are owned by Duke University.
# Developed by Ming-Yang Ho and Alberto Bartesaghi at the Bartesaghi Lab,
# Department of Computer Science, Trinity College of Arts & Sciences, Duke
# University.
# ========================================================================
from fulltilt.models.FullTilt import build_fulltilt  # noqa: F401
from fulltilt.models import registry


def build_model(args):
    if args.modelname not in registry.MODULE_BUILD_FUNCS._module_dict:
        raise ValueError(f'Model {args.modelname} cannot be found.')
    build_func = registry.MODULE_BUILD_FUNCS.get(args.modelname)
    model = build_func(args)
    return model
