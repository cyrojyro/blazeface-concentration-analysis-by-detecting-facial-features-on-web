export let status = {
  yaw: 0,
  roll: 0,
  pitch: 0,
  notDetected: false,
  turned: false,
  turnedFactor: 0,
  bowed: false,
  bowedFactor: 0,
  eyesClosed: false,
  eyesClosedFactor: 0,
};

const weights = { notDetected: 50, turned: 30, bowed: 40, eyesClosed: 50 };
const eyeFactor = 5.8; // higher -> need to close eye harder to trigger true
const turnFactor = 3.5; // higher -> need more turn to trigger true
const bowFactor = -0.1; // higher -> need more bow to trigger true
const eyeTurnCorrection = 2.5;

export function analyze(detection, landmarks, angle) {
  status.notDetected = detection ? false : true;

  if (landmarks) {
    status = { ...status, ...analyzeLandmark(landmarks) };
    status.pitch = angle[0][0].toFixed(3);
    status.yaw = angle[0][1].toFixed(3);
    status.roll = angle[0][2].toFixed(3);
  }

  // weighted sum of score to produce overall score.
  return Object.keys(weights).reduce((acc, key) => {
    const returnVal = status[key] ? acc + status[key] * weights[key] : acc;
    return returnVal;
  }, 0);
}

function calcDist(p1, p2) {
  return Math.sqrt(Math.pow(p1._x - p2._x, 2) + Math.pow(p1._y - p2._y, 2));
}

function diffBigger(l1, l2, ratio) {
  if (l1 < l2) {
    let t = l1;
    l1 = l2;
    l2 = t;
  }
  return [l2 * ratio < l1, `${(l1 / l2).toFixed(2)}>${turnFactor}`];
}

function analyzeLandmark(landmarks) {
  // turned face
  const lcheek = calcDist(landmarks[33], landmarks[3]);
  const rcheek = calcDist(landmarks[33], landmarks[13]);
  const [turned, turnedFactor] = diffBigger(lcheek, rcheek, turnFactor);

  // bowed face
  const facehigh = (landmarks[1]._y + landmarks[15]._y) / 2;
  const eyelow = (landmarks[39]._y + landmarks[42]._y) / 2;
  const distance =
    (calcDist(landmarks[0], landmarks[1]) +
      calcDist(landmarks[15], landmarks[16])) /
    2;
  const bowed = facehigh < eyelow + distance * -bowFactor;
  const bowedFactor = `${(-(facehigh - eyelow) / distance).toFixed(
    2
  )}>${bowFactor}`;

  // eyes closed
  const leyeHeight = calcDist(landmarks[38], landmarks[40]);
  const reyeHeight = calcDist(landmarks[43], landmarks[47]);
  const avgeyeHeight = (leyeHeight + reyeHeight) / 2;
  const leyeWidth = calcDist(landmarks[36], landmarks[39]);
  const reyeWidth = calcDist(landmarks[42], landmarks[45]);
  const avgeyeWidth = (leyeWidth + reyeWidth) / 2;
  const cheekRatio =
    Math.pow(Math.max(lcheek, rcheek) - Math.min(lcheek, rcheek), 2) /
    Math.pow(Math.max(lcheek, rcheek), 2);
  const eyesClosed =
    avgeyeWidth / avgeyeHeight + eyeTurnCorrection * cheekRatio >= eyeFactor;
  const eyesClosedFactor = `${(
    avgeyeWidth / avgeyeHeight +
    eyeTurnCorrection * cheekRatio
  ).toFixed(2)}>${eyeFactor}`;

  return {
    turned,
    turnedFactor,
    bowed,
    bowedFactor,
    eyesClosed,
    eyesClosedFactor,
  };
}
