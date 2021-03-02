import * as tfjs from "@tensorflow/tfjs";

class LandmarkModel {
  constructor(imgW, imgH) {
    this.model = null;
    this.imgW = imgW;
    this.imgH = imgH;
  }

  async loadFromUri(url) {
    this.model = await tfjs.loadGraphModel(url);
  }

  async predict(bbox, img) {
    const croppedFace = this.cropFace(bbox, img);
    if (croppedFace === undefined) return [undefined, undefined];

    const [angles, _rawLandmarks] = this.model.execute(croppedFace, [
      "Identity_1",
      "Identity_2",
    ]);

    const toCopy = [angles, _rawLandmarks];
    const [angle, rawLandmarks] = await Promise.all(
      toCopy.map(async (item) => {
        const arr = await item.array();
        return arr;
      })
    );

    const landmarks = convertLandmark(rawLandmarks, bbox);
    tfjs.dispose([croppedFace, angles, landmarks]);
    return [angle, landmarks];
  }

  cropFace(bbox, img) {
    if (bbox === undefined) return undefined;
    return tfjs.image.cropAndResize(
      img,
      [
        [
          bbox[1] / this.imgH,
          bbox[0] / this.imgW,
          bbox[3] / this.imgH,
          bbox[2] / this.imgW,
        ],
      ],
      [0],
      [160, 160]
    );
  }
}

export function convertLandmark(landmarks, box) {
  if (box === undefined) return undefined;

  let landmarkObj = new Array();
  for (let i = 0; i < 136; ++i) {
    if (!(i % 2)) landmarkObj[Math.floor(i / 2)] = new Object();

    let o =
      i % 2
        ? { key: "_y", offset: box[1], mult: box[3] - box[1] }
        : { key: "_x", offset: box[0], mult: box[2] - box[0] };
    landmarkObj[Math.floor(i / 2)][o.key] = o.offset + landmarks[0][i] * o.mult;
  }
  return landmarkObj;
}

export const landmarkModel = new LandmarkModel(640, 480);
