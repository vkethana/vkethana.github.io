(function () {
  var win = document.getElementById("site-window");
  var desktopIcon = document.getElementById("desktop-icon");
  var taskBtn = document.getElementById("task-btn");
  var minBtn = document.getElementById("win-minimize");
  var maxBtn = document.getElementById("win-maximize");
  var closeBtn = document.getElementById("win-close");

  if (!win || !desktopIcon || !taskBtn || !minBtn || !maxBtn || !closeBtn) return;

  var STORAGE_KEY = "vk-xp-window-state";

  function getState() {
    return win.getAttribute("data-window-state") || "normal";
  }

  function setState(state) {
    win.setAttribute("data-window-state", state);
    document.body.setAttribute("data-window-state", state);

    var isClosed = state === "closed";
    var isMinimized = state === "minimized";
    var isMaximized = state === "maximized";
    var isHidden = isClosed || isMinimized;

    win.hidden = isHidden;
    desktopIcon.hidden = !isClosed;
    taskBtn.hidden = isClosed;
    taskBtn.classList.toggle("is-active", !isHidden && !isClosed);
    taskBtn.classList.toggle("is-minimized", isMinimized);

    maxBtn.classList.toggle("is-restore", isMaximized);
    maxBtn.title = isMaximized ? "Restore Down" : "Maximize";
    maxBtn.setAttribute("aria-label", maxBtn.title);

    try {
      sessionStorage.setItem(STORAGE_KEY, state);
    } catch (e) {}
  }

  function restore() {
    var prev = win.getAttribute("data-prev-state");
    setState(prev === "maximized" ? "maximized" : "normal");
  }

  function openWindow() {
    setState("normal");
  }

  minBtn.addEventListener("click", function () {
    if (getState() === "maximized") {
      win.setAttribute("data-prev-state", "maximized");
    } else {
      win.setAttribute("data-prev-state", "normal");
    }
    setState("minimized");
  });

  maxBtn.addEventListener("click", function () {
    if (getState() === "maximized") {
      setState("normal");
    } else if (getState() === "minimized") {
      setState("maximized");
    } else {
      setState("maximized");
    }
  });

  closeBtn.addEventListener("click", function () {
    setState("closed");
  });

  taskBtn.addEventListener("click", function () {
    if (getState() === "minimized") {
      restore();
    } else if (getState() === "closed") {
      openWindow();
    } else if (getState() === "maximized" || getState() === "normal") {
      win.setAttribute("data-prev-state", getState());
      setState("minimized");
    }
  });

  desktopIcon.addEventListener("click", openWindow);
  desktopIcon.addEventListener("dblclick", openWindow);

  // Double-click title bar to maximize/restore (XP behavior)
  var titleBar = win.querySelector(".title-bar");
  if (titleBar) {
    titleBar.addEventListener("dblclick", function (e) {
      if (e.target.closest(".window-controls") || e.target.closest("a")) return;
      maxBtn.click();
    });
  }

  var saved = null;
  try {
    saved = sessionStorage.getItem(STORAGE_KEY);
  } catch (e) {}

  if (saved === "closed" || saved === "minimized" || saved === "maximized" || saved === "normal") {
    if (saved === "minimized") {
      win.setAttribute("data-prev-state", "normal");
    }
    setState(saved);
  } else {
    setState("normal");
  }
})();
