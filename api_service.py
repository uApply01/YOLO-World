import os.path as osp
from typing import List, Dict

import cv2
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import torch
from mmengine.config import Config, DictAction
from mmengine.runner.amp import autocast
from mmengine.dataset import Compose
from mmengine.utils import ProgressBar
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg
import base64
from io import BytesIO
import tempfile
import os

app = FastAPI()

# 模型配置和初始化
CONFIG_FILE = "configs/pretrain/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
CHECKPOINT = "weights/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth"

# 初始化模型（全局变量）
cfg = Config.fromfile(CONFIG_FILE)
cfg.work_dir = osp.join('./work_dirs')
cfg.load_from = CHECKPOINT
model = init_detector(cfg, checkpoint=CHECKPOINT, device='cuda:0')
test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
# test_pipeline_cfg[0].type = 'mmdet.LoadImageFromNDArray'
test_pipeline = Compose(test_pipeline_cfg)

class PredictRequest(BaseModel):
    labels: str
    image: str  # base64 encoded image

class DetectionResult(BaseModel):
    bbox: List[float]
    label: str
    score: float

@app.post("/predict", response_model=List[DetectionResult])
async def predict(request: PredictRequest):
    # Create a temporary file
    temp_file = None
    try:
        # Decode base64 image
        try:
            image_data = base64.b64decode(request.image)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Invalid image data")
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid image data")

        # Save image to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        cv2.imwrite(temp_file.name, image)
        
        # Prepare text prompts
        texts = [[t.strip()] for t in request.labels.split(',')] + [[' ']]

        # Reparameterize model with new texts
        model.reparameterize(texts)

        # Prepare data for inference using file path
        print(temp_file.name)
        data_info = dict(img_id=0, img_path=temp_file.name, texts=texts)
        data_info = test_pipeline(data_info)
        data_batch = dict(
            inputs=data_info['inputs'].unsqueeze(0),
            data_samples=[data_info['data_samples']]
        )

        # Inference with thresholds
        with autocast(enabled=False), torch.no_grad():
            output = model.test_step(data_batch)[0]
            pred_instances = output.pred_instances
            pred_instances = pred_instances[pred_instances.scores.float() > 0.3]

        # Apply top-k filtering
        if len(pred_instances.scores) > 100:
            indices = pred_instances.scores.float().topk(100)[1]
            pred_instances = pred_instances[indices]

        # Convert to numpy and prepare response
        pred_instances = pred_instances.cpu().numpy()
        results = []
        for box, label_idx, score in zip(
            pred_instances['bboxes'],
            pred_instances['labels'],
            pred_instances['scores']
        ):
            results.append(DetectionResult(
                bbox=box.tolist(),
                label=texts[label_idx][0],
                score=float(score)
            ))

        return results
    finally:
        # Clean up temporary file
        if temp_file is not None:
            try:
                os.unlink(temp_file.name)
            except Exception:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
