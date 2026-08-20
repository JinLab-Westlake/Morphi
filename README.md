

<h1>Morphi</h1>

<img
  src="./docs/images/Picture1.png"
  align="right"
  width="350"
  alt="Multi-dimensional organoid phenotype analysis software"
/>

AI-powered organoid analysis software integrating:
1. z-stack projection
2. fluorescence quantification
3. instance segmentation
4. organoid viability assessment

[Documentation](https://xxx.github.io/Morphi/) |
[Datasets](链接) |
[Paper](链接)




<h2>Interface</h2>

<p align="center">
  <img src="docs/images/morphi_interface.png" width="600">
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

<p>Create a virtual environment:</p>

<pre><code>python -m venv .venv</code></pre>


<h4>For different operating systems:</h4>
<h4>1. Linux / macOS</h4>

<pre><code>source .venv/bin/activate</code></pre>

<h4>2. Windows (Command Prompt / PowerShell)</h4>

<pre><code># Windows - Command Prompt
.venv\Scripts\activate

# Windows - PowerShell
.venv\Scripts\Activate.ps1</code></pre>


<p>Install the required dependencies:</p>

<pre><code>pip install -r requirements.txt</code></pre>

<p>Run Morphi:</p>

<pre><code>python morphi/main.py</code></pre>


<h2>Usage</h2>
<p>
  For detailed usage instructions of software, see the
  <a href="https://xxx.github.io/Morphi/">Documentation</a>.
</p>





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
  <a href="YOUR_URL" target="_blank">Morphi-checkpoints</a>.
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
