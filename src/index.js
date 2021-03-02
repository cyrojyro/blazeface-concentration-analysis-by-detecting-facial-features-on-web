import * as tfjs from "@tensorflow/tfjs";
import { status, analyze } from "./analysis.js";
import { landmarkModel } from "./landmark.js";
import { detectorModel } from "./detector.js";
tfjs.enableProdMode();

let score = 0;
export let frames = 0;
const stop = document.getElementById("stopButton");
const resume = document.getElementById("resumeButton");
const video = document.getElementById("video");
const vidW = 640;
const vidH = 480;

stop.onclick = onStop;
resume.onclick = onResume;

let task = true;
function onStop() {
  task = false;
  video.pause();
}
function onResume() {
  task = true;
  video.load();
}

Promise.all([
  landmarkModel.loadFromUri("../dist/models-tfjs/keypoints_tfjs/model.json"),
  detectorModel.loadFromUri(
    "../dist/models-tfjs/detector_crafted_q/model.json"
  ),
]).then(startVideo);

function startVideo() {
  navigator.getUserMedia(
    { video: { width: vidW, height: vidH } },
    (stream) => (video.srcObject = stream),
    (err) => console.error(err)
  );
}

video.addEventListener("play", async () => {
  console.log(`Backend: ${tfjs.getBackend()}`);
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");
  document.body.append(canvas);
  setTimeout(async function faceAnalysis() {
    if (!task) return;

    const timefd1 = performance.now();
    const pixel = tfjs.browser.fromPixels(video);
    const img = pixel.reshape([-1, vidH, vidW, 3]);
    const detectImg = tfjs.image.resizeBilinear(img, [128, 128]);
    const [bbox, conf] = await detectorModel.predict(detectImg);
    const timefd2 = performance.now();

    const timelm1 = performance.now();
    const [angle, landmark] = await landmarkModel.predict(bbox, img);
    const timelm2 = performance.now();

    score = analyze(bbox, landmark, angle);
    drawAll(canvas, ctx, bbox, conf, landmark, score);

    // time checker
    frames = frames + 1;
    if (frames % 10 === 0)
      console.log(
        `${frames}: fd ${(timefd2 - timefd1).toFixed(3)}ms lm ${(
          timelm2 - timelm1
        ).toFixed(3)}ms`
      );

    tfjs.dispose([landmark, detectImg, angle, pixel, img]);
    setTimeout(faceAnalysis, 0);
  }, 0);
});

function drawAll(canvas, ctx, bbox, conf, landmarkObj, score) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.fillStyle = "#FF0000";
  ctx.strokeStyle = "#FF0000";
  ctx.font = "30px Arial";
  ctx.lineWidth = "4";

  if (bbox !== undefined) {
    ctx.font = "30px Arial";
    ctx.beginPath();
    ctx.rect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]);
    ctx.fillText(conf.toFixed(2), bbox[0] + 15, bbox[1] + 30);
    ctx.stroke();
    for (let i = 0; i < 68; ++i) {
      ctx.fillRect(landmarkObj[i]["_x"], landmarkObj[i]["_y"], 4, 4);
    }
  }
  drawInfo(ctx, score);
}

function drawInfo(ctx, score) {
  ctx.fillText("score: " + score, 30, 50);
  ctx.font = "18px Arial";
  var lines = JSON.stringify(status, null, 2).split("\n");
  for (var j = 0; j < lines.length; j++)
    ctx.fillText(lines[j], 10, 240 + j * 20);
  ctx.font = "14px Arial";
  ctx.fillText(JSON.stringify(tfjs.memory()), 20, 470);
}
