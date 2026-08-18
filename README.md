

<h1>Morphi</h1>

<img
  src="./docs/images/Picture1.png"
  align="right"
  width="350"
  alt="Multi-dimensional organoid phenotype analysis platform"
/>

AI-powered organoid analysis platform integrating:
1. z-stack projection
2. fluorescence quantification
3. instance segmentation
4. organoid viability assessment

[Documentation](https://xxx.github.io/Morphi/) |
[Datasets](链接) |
[Paper](链接)




<h2>Interface</h2>

<p align="left">
  <img src="docs/images/morphi_interface.png" width="500">
</p>


<h2>Installation</h2>

<h3>macOS application</h3>

<p>
  Download the latest <code>Morphi.app</code> from the
  <a href="https://github.com/xxx/Morphi/releases">Releases</a> page
  and move it to the <code>Applications</code> folder.
</p>

<p>
  Morphi can then be launched directly from the Applications folder.
</p>

<h4>Optional: Terminal launcher</h4>

<p>
  To launch Morphi from Terminal using the <code>morphi</code> command,
  install the command-line launcher:
</p>

<pre><code>sudo mkdir -p /usr/local/bin

sudo tee /usr/local/bin/morphi &gt; /dev/null &lt;&lt;'EOF'
#!/bin/sh
open "/Applications/Morphi.app"
EOF

sudo chmod +x /usr/local/bin/morphi</code></pre>

<p>
  Morphi can then be launched from any directory:
</p>

<pre><code>morphi</code></pre>

<h2>Usage</h2>

<p>
  Launch Morphi from the Applications folder or from Terminal:
</p>

<pre><code>morphi</code></pre>

<p>
  For detailed usage instructions, see the
  <a href="https://xxx.github.io/Morphi/">Documentation</a>.
</p>




<h2>Installation</h2>

<p>
  Morphi can be installed as a standalone macOS application or run directly from source.
</p>

<h3>Option 1: macOS application</h3>

<p>
  Download the latest <code>Morphi.app</code> from the
  <a href="https://github.com/JinLab-Westlake/Morphi/releases">Releases</a> page.
</p>

<p>
  Move <code>Morphi.app</code> to the <code>Applications</code> folder.
</p>

<p>
  Morphi is packaged as a standalone application, so no separate Python environment
  or dependency installation is required.
</p>

<p>
  Once installed, Morphi can be launched from any directory using:
</p>

<pre><code>morphi</code></pre>

<p>
  Morphi uses application-relative resource paths, so it does not depend on the
  current working directory.
</p>

<h3>Option 2: Run from source</h3>

<p>Clone the repository:</p>

<pre><code>git clone https://github.com/JinLab-Westlake/Morphi.git
cd Morphi</code></pre>

<p>Create and activate a virtual environment:</p>

<pre><code>python -m venv .venv
source .venv/bin/activate</code></pre>

<p>Install the required dependencies:</p>

<pre><code>pip install -r requirements.txt</code></pre>

<p>Run Morphi:</p>

<pre><code>python morphi/main.py</code></pre>


<h2>Model weight</h2>
<p>The
  <a href="https://github.com/JinLab-Westlake/Morphi-models" target="_blank">
    DoANet
  </a> and
  <a href="https://github.com/JinLab-Westlake/Morphi-models" target="_blank">
    Instance-Segmentation
  </a>model export are based on ONNX format.
</p>

<p>
  The trained model weights are available for download from
  <a href="YOUR_URL" target="_blank">Morphi-weight</a>.
</p>

<p>
  Model training code can be found on
  <a href="https://github.com/JinLab-Westlake/Morphi-models" target="_blank">
    Morphi-models
  </a>.
</p>



<h2>Citation</h2>

Citation information after publication of the paper.

<h2>License</h2>

Code license
