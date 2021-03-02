const { app, BrowserWindow } = require("electron");

function createWindow() {
  // Create the browser window.
  const win = new BrowserWindow({
    width: 1280,
    height: 960,
    webPreferences: {
      nodeIntegration: true,
      backgroundThrottling: false,
    },
  });

  // and load the index.html of the app.
  win.loadFile("index.html");
  win.webContents.openDevTools();
}

app.whenReady().then(createWindow);
