from slowfast.config.defaults import get_cfg
from slowfast.models import build_model
from slowfast.utils.checkpoint import load_checkpoint

def test():
    cfg = get_cfg()
    cfg.merge_from_file("configs/Kinetics/SLOWFAST_8x8_R50.yaml")
    cfg.NUM_GPUS = 0
    cfg.MODEL.NUM_CLASSES = 400
    cfg.TEST.CHECKPOINT_FILE_PATH = "slowfast/SLOWFAST_8x8_R50.pyth"

    model = build_model(cfg)
    load_checkpoint(cfg.TEST.CHECKPOINT_FILE_PATH, model)
    model.eval()

    print("✅ SlowFast model loaded successfully")

if __name__ == "__main__":
    test()