param(
    [string]$VenvName = ".venv_jos3"
)

Write-Host "==== Resetando ambiente virtual: $VenvName ===="

# Caminho absoluto do ambiente virtual
$venvPath = Join-Path -Path $PSScriptRoot -ChildPath $VenvName

# 1. Remove o venv antigo (se existir)
if (Test-Path $venvPath) {
    Write-Host "Removendo venv antigo..."
    Remove-Item $venvPath -Recurse -Force
}

# 2. Cria o novo venv
Write-Host "Criando novo ambiente virtual..."
python -m venv $venvPath

# Verifica se deu certo
if (!(Test-Path (Join-Path $venvPath "Scripts\python.exe"))) {
    Write-Error "Falha ao criar o venv. Garanta que 'python' está no PATH."
    exit 1
}

# 3. Ativa o ambiente
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
Write-Host "Ativando o novo venv..."
& $activateScript

# 4. Atualiza pip
Write-Host "Atualizando pip..."
python -m pip install --upgrade pip

# 5. Instala pacotes do requirements.txt
if (Test-Path "requirements.txt") {
    Write-Host "Instalando pacotes do requirements.txt..."
    python -m pip install -r requirements.txt
} else {
    Write-Error "requirements.txt não encontrado na pasta do projeto!"
    exit 1
}

Write-Host "==== Ambiente virtual configurado com sucesso! ===="
Write-Host "Para usar depois, ative com:"
Write-Host ".\$VenvName\Scripts\Activate.ps1"