$password = ConvertTo-SecureString "207dajaiting" -AsPlainText -Force
$username = "lab207"
$server = "192.168.1.126"
$remotePath = "/home/lab207/ljy/pythonProject3"

# Create remote directory first
$sshCommand = "ssh -o StrictHostKeyChecking=no $username@$server 'mkdir -p $remotePath'"
Write-Host "Creating remote directory..."

# Upload files using scp
Write-Host "Uploading files to server..."
$scpCommand = "scp -o StrictHostKeyChecking=no -r * $username@${server}:$remotePath/"
Invoke-Expression $scpCommand
