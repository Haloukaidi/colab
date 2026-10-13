"""Smoke inference API after training and saving an actual Lite50 checkpoint."""
import argparse
from PIL import Image
from lite50 import build_lite50_image_model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image",required=True)
    p.add_argument("--checkpoint",required=True,help="TRAINED Lite50 weights")
    p.add_argument("--text",default="dog")
    p.add_argument("--box",nargs=4,type=float,help="Normalized cx cy width height")
    p.add_argument("--point",nargs=2,type=float,help="Pixel x y")
    p.add_argument("--device",default="cuda")
    a=p.parse_args()
    from sam3.model.sam3_image_processor import Sam3Processor
    m=build_lite50_image_model(lite_checkpoint_path=a.checkpoint,
                               initialize_text_from_dense=False,device=a.device)
    proc=Sam3Processor(m,device=a.device)
    state=proc.set_image(Image.open(a.image).convert("RGB"))
    state=proc.set_text_prompt(a.text,state)
    if a.box:
        state=proc.add_geometric_prompt(a.box,True,state)
    if a.point:
        state=proc.add_point_prompt(a.point,1,state)
    print("masks",state["masks"].shape,"scores",state["scores"])

if __name__=="__main__":
    main()
