import launch

if not launch.is_installed("rapidfuzz"):
    launch.run_pip("install rapidfuzz>=3.0", "rapidfuzz for sd-webui-prompt-enhancer")
