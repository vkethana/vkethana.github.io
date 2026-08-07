(function () {
  var filter = document.getElementById("tag-filter");
  var list = document.getElementById("post-list");
  var empty = document.getElementById("tag-filter-empty");
  if (!filter || !list) return;

  var chips = filter.querySelectorAll(".tag-chip");
  var cards = list.querySelectorAll(".post-card");

  function setFilter(tag) {
    var visible = 0;
    cards.forEach(function (card) {
      var tags = (card.getAttribute("data-tags") || "").split("|").filter(Boolean);
      var show = tag === "all" || tags.indexOf(tag) !== -1;
      card.hidden = !show;
      if (show) visible += 1;
    });

    chips.forEach(function (chip) {
      var active = chip.getAttribute("data-tag") === tag;
      chip.classList.toggle("is-active", active);
      chip.setAttribute("aria-pressed", active ? "true" : "false");
    });

    if (empty) empty.hidden = visible > 0;

    try {
      if (tag === "all") {
        history.replaceState(null, "", location.pathname + location.search);
      } else {
        history.replaceState(null, "", "#tag=" + encodeURIComponent(tag));
      }
    } catch (e) {}
  }

  filter.addEventListener("click", function (e) {
    var chip = e.target.closest(".tag-chip");
    if (!chip || !filter.contains(chip)) return;
    setFilter(chip.getAttribute("data-tag") || "all");
  });

  var initial = "all";
  var hash = location.hash || "";
  var match = hash.match(/^#tag=(.+)$/);
  if (match) {
    try {
      initial = decodeURIComponent(match[1]);
    } catch (e) {}
  }
  setFilter(initial);
})();
