# Install nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash

# Reload shell
source ~/.bashrc

# Install Node 20
nvm install 20
nvm use 20
node --version  # should show v20.x

# Now install Claude Code
npm install -g @anthropic-ai/claude-code

pip install "huggingface_hub[hf_transfer]"


