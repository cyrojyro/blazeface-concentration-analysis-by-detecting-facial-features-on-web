import * as tfjs from "@tensorflow/tfjs";

class DetectorModel {
  constructor(imgW, imgH) {
    this.model = null;
    this.imgW = imgW;
    this.imgH = imgH;
    this.anchors = generate_anchors(0.2, 0.9, [2, 6], [16, 8]);
  }

  async loadFromUri(url) {
    this.model = await tfjs.loadGraphModel(url);
  }

  async predict(image) {
    const normalized = normalizeImage(image);
    const preds = this.model.predict(normalized);
    const pred = tfjs.squeeze(preds, 0);
    const [confs, bboxes] = prediction_to_bbox(
      pred,
      this.anchors,
      this.imgW,
      this.imgH
    );
    const [selectedBox, conf] = await nonMaximalSuppression(bboxes, confs, 0.3);
    tfjs.dispose([normalized, preds, pred, confs, bboxes]);
    return [selectedBox, conf];
  }
}

function normalizeImage(image) {
  const normalized = tfjs.tidy(() => {
    return tfjs.sub(tfjs.div(image, 127.5), 1.0);
  });
  return normalized;
}

function generate_anchors(sMin, sMax, anchorNum, cellSize) {
  let totalNum = anchorNum.reduce((acc, val) => acc + val, 0);
  let anchors = new Array();
  let cellAcc = 0;

  for (let iter = 0; iter < anchorNum.length; iter++) {
    let cells = cellSize[iter];
    for (let y = 0; y < cells; y++) {
      for (let x = 0; x < cells; x++) {
        for (let order = 0; order < anchorNum[iter]; order++) {
          let scale =
            sMin + ((sMax - sMin) / (totalNum - 1)) * (order + cellAcc);
          anchors.push([(x + 0.5) / cells, (y + 0.5) / cells, scale, scale]);
        }
      }
    }
    cellAcc += anchorNum[iter];
  }
  return tfjs.tensor(anchors);
}

function prediction_to_bbox(prediction, anchors, imgW, imgH) {
  const result = tfjs.tidy(() => {
    const anchorScales = tfjs.slice(anchors, [0, 2], [anchors.shape[0], 2]);
    const anchorCoords = tfjs.slice(anchors, [0, 0], [anchors.shape[0], 2]);
    const predScales = tfjs.slice(prediction, [0, 3], [prediction.shape[0], 2]);
    const predCoords = tfjs.slice(prediction, [0, 1], [prediction.shape[0], 2]);
    const predConfs = tfjs
      .slice(prediction, [0, 0], [prediction.shape[0], 1])
      .squeeze(-1);
    const imgSize = tfjs.tensor([imgW, imgH]);
    const center = tfjs
      .add(tfjs.mul(predCoords, anchorScales), anchorCoords)
      .mul(imgSize);
    const widthHeight = tfjs
      .mul(anchorScales, tfjs.exp(predScales))
      .mul(imgSize);

    const topLeft = tfjs.sub(center, tfjs.div(widthHeight, 2));
    const downRight = tfjs.add(center, tfjs.div(widthHeight, 2));
    const coords = tfjs.concat([topLeft, downRight], -1);
    return [predConfs, coords];
  });
  return result;
}

async function nonMaximalSuppression(bboxes, confs, min_conf) {
  const topk = confs.topk(1);

  const toCopy = [topk["values"], topk["indices"]];
  const [[maxConf], [maxArg]] = await Promise.all(
    toCopy.map(async (item) => {
      const arr = await item.array();
      return arr;
    })
  );

  if (maxConf < min_conf) {
    tfjs.dispose(topk);
    return [undefined, undefined];
  }

  let bbox = await bboxes.array();
  bbox = bbox[maxArg];
  tfjs.dispose(topk);
  return [bbox, maxConf];
}

export const detectorModel = new DetectorModel(640, 480);
