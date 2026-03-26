import paramiko
import os
from scp import SCPClient

def upload_directory(ssh_client, local_path, remote_path):
    """递归上传整个目录"""
    with SCPClient(ssh_client.get_transport()) as scp:
        print(f"Uploading {local_path} to {remote_path}...")
        scp.put(local_path, remote_path=remote_path, recursive=True)
        print("Upload completed!")

def main():
    # 服务器信息
    hostname = '192.168.1.126'
    username = 'lab207'
    password = '207dajaiting'
    remote_base_path = '/home/lab207/ljy'
    remote_dir = 'pythonProject3'

    # 当前目录
    local_path = os.getcwd()

    try:
        # 创建 SSH 客户端
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        print(f"Connecting to {hostname}...")
        ssh.connect(hostname, username=username, password=password)
        print("Connected successfully!")

        # 创建远程目录
        remote_full_path = f'{remote_base_path}/{remote_dir}'
        print(f"Creating remote directory: {remote_full_path}")
        stdin, stdout, stderr = ssh.exec_command(f'mkdir -p {remote_full_path}')
        stdout.channel.recv_exit_status()  # 等待命令执行完成

        # 上传文件
        print("Starting upload...")
        with SCPClient(ssh.get_transport(), progress=lambda filename, size, sent:
                      print(f"Uploading: {filename} - {sent}/{size} bytes")) as scp:
            # 上传当前目录下的所有文件
            for item in os.listdir(local_path):
                if item == 'upload_to_server.py' or item == 'upload_script.ps1':
                    continue  # 跳过上传脚本本身

                local_item = os.path.join(local_path, item)
                remote_item_path = f'{remote_full_path}/'

                if os.path.isfile(local_item):
                    print(f"Uploading file: {item}")
                    scp.put(local_item, remote_path=remote_item_path)
                elif os.path.isdir(local_item):
                    print(f"Uploading directory: {item}")
                    scp.put(local_item, remote_path=remote_item_path, recursive=True)

        print("\n✓ All files uploaded successfully!")

        # 验证上传
        print("\nVerifying uploaded files...")
        stdin, stdout, stderr = ssh.exec_command(f'ls -lh {remote_full_path}')
        print(stdout.read().decode())

        ssh.close()

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0

if __name__ == '__main__':
    main()
