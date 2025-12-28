const fileInput = document.getElementById("fileInput");
const traceSelect = document.getElementById("traceSelect");
const treeContainer = document.getElementById("treeContainer");
const detailContent = document.getElementById("detailContent");
const metricsDiv = document.getElementById("metrics");
const graphSvg = d3.select("#graph");

let traces = [];

/* -------------------------
   File loading
------------------------- */

async function loadTracesFromText(text) {
  const parsed = [];

  text
    .split("\n")
    .filter(Boolean)
    .forEach((line, idx) => {
      try {
        parsed.push(JSON.parse(line));
      } catch (err) {
        console.error(`Skipping invalid JSON on line ${idx + 1}`, err);
      }
    });

  if (!parsed.length) {
    traces = [];
    traceSelect.innerHTML = "";
    treeContainer.innerHTML = "";
    metricsDiv.innerHTML = "";
    graphSvg.selectAll("*").remove();
    detailContent.textContent =
      "No traces loaded. Select a .jsonl file to begin.";
    return;
  }

  traces = parsed;
  traceSelect.innerHTML = "";
  traces.forEach((t, i) => {
    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = `${i + 1}: ${t.question.slice(0, 60)}...`;
    traceSelect.appendChild(opt);
  });

  renderTrace(0);
}

fileInput.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const text = await file.text();
  loadTracesFromText(text);
});

async function loadDefaultDataset() {
  try {
    const res = await fetch("trees.jsonl");
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    const text = await res.text();
    loadTracesFromText(text);
  } catch (err) {
    console.error("Failed to load default dataset", err);
    detailContent.textContent =
      "Default dataset (trees.jsonl) could not be loaded.\n" +
      "Please choose a .jsonl file manually.";
  }
}

// Load trees.jsonl by default, but still allow manual uploads.
loadDefaultDataset();

traceSelect.addEventListener("change", (e) => {
  renderTrace(+e.target.value);
});

/* -------------------------
   Tree Explorer
------------------------- */

function renderTree(tree, parentEl) {
  const li = document.createElement("li");
  li.className = `tree-node ${tree.type}`;
  li.textContent =
    tree.type === "node"
      ? "Node"
      : "Leaf";

  li.onclick = (e) => {
    e.stopPropagation();
    detailContent.textContent = JSON.stringify(tree, null, 2);
  };

  parentEl.appendChild(li);

  if (tree.type === "node") {
    const ul = document.createElement("ul");
    li.appendChild(ul);

    tree.subtasks.forEach((st) => {
      renderTree(st.execution, ul);
    });
  }
}

/* -------------------------
   Metrics
------------------------- */

function computeMetrics(tree) {
  let nodes = 0;
  let leaves = 0;
  let maxDepth = 0;

  function dfs(t, depth) {
    maxDepth = Math.max(maxDepth, depth);
    if (t.type === "node") {
      nodes++;
      t.subtasks.forEach((s) => dfs(s.execution, depth + 1));
    } else {
      leaves++;
    }
  }

  dfs(tree, 1);

  return { nodes, leaves, maxDepth };
}

function renderMetrics(tree) {
  const m = computeMetrics(tree);
  metricsDiv.innerHTML = `
    <div>Nodes: ${m.nodes}</div>
    <div>Leaves: ${m.leaves}</div>
    <div>Max depth: ${m.maxDepth}</div>
  `;
}

/* -------------------------
   Graph View (DAG)
------------------------- */

function buildGraph(tree) {
  let nodes = [];
  let links = [];
  let idCounter = 0;

  function walk(t, parentId = null) {
    const myId = idCounter++;
    nodes.push({
      id: myId,
      type: t.type
    });

    if (parentId !== null) {
      links.push({ source: parentId, target: myId });
    }

    if (t.type === "node") {
      t.subtasks.forEach((s) => walk(s.execution, myId));
    }
  }

  walk(tree);
  return { nodes, links };
}

function renderGraph(tree) {
  graphSvg.selectAll("*").remove();

  const { nodes, links } = buildGraph(tree);

  const simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id(d => d.id).distance(50))
    .force("charge", d3.forceManyBody().strength(-150))
    .force("center", d3.forceCenter(300, 200));

  const link = graphSvg.append("g")
    .selectAll("line")
    .data(links)
    .enter()
    .append("line")
    .attr("stroke", "#aaa");

  const node = graphSvg.append("g")
    .selectAll("circle")
    .data(nodes)
    .enter()
    .append("circle")
    .attr("r", d => d.type === "node" ? 8 : 5)
    .attr("fill", d => d.type === "node" ? "#2563eb" : "#16a34a")
    .call(d3.drag()
      .on("start", dragstarted)
      .on("drag", dragged)
      .on("end", dragended)
    );

  node.on("click", (_, d) => {
    detailContent.textContent = JSON.stringify(d, null, 2);
  });

  simulation.on("tick", () => {
    link
      .attr("x1", d => d.source.x)
      .attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x)
      .attr("y2", d => d.target.y);

    node
      .attr("cx", d => d.x)
      .attr("cy", d => d.y);
  });

  function dragstarted(event) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    event.subject.fx = event.subject.x;
    event.subject.fy = event.subject.y;
  }

  function dragged(event) {
    event.subject.fx = event.x;
    event.subject.fy = event.y;
  }

  function dragended(event) {
    if (!event.active) simulation.alphaTarget(0);
    event.subject.fx = null;
    event.subject.fy = null;
  }
}

/* -------------------------
   Main render
------------------------- */

function renderTrace(index) {
  const trace = traces[index];
  if (!trace) return;

  treeContainer.innerHTML = "";
  const ul = document.createElement("ul");
  treeContainer.appendChild(ul);
  renderTree(trace.tree, ul);

  renderMetrics(trace.tree);
  renderGraph(trace.tree);

  detailContent.textContent = JSON.stringify(trace.tree, null, 2);
}

