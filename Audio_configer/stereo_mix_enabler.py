import subprocess
import time

def enable_stereo_mix(set_as_default=False):
    try:
        # Use PowerShell to manage audio devices
        # First, check if AudioDeviceCmdlets module is installed
        check_module = 'Get-Module -ListAvailable AudioDeviceCmdlets'
        result = subprocess.run(['powershell', '-Command', check_module], capture_output=True, text=True)
        
        if not result.stdout.strip():
            print("Installing AudioDeviceCmdlets module...")
            install_cmd = 'Install-Module -Name AudioDeviceCmdlets -Force -Scope CurrentUser'
            subprocess.run(['powershell', '-Command', install_cmd], capture_output=True)
            time.sleep(2)  # Wait for installation
        
        # List all audio devices
        list_cmd = 'Get-AudioDevice -List | Format-List'
        devices = subprocess.run(['powershell', '-Command', list_cmd], capture_output=True, text=True)
        print("\nAvailable audio devices:")
        print(devices.stdout)
        
        # Find and enable Stereo Mix
        enable_cmd = '''
        $devices = Get-AudioDevice -List
        $stereoMix = $devices | Where-Object { $_.Name -like "*Stereo Mix*" }
        if ($stereoMix) {
            Write-Output "Found Stereo Mix: $($stereoMix.Name)"
            # Enable the device
            Set-AudioDevice -ID $stereoMix.ID
            Write-Output "Stereo Mix enabled"
        } else {
            Write-Output "Stereo Mix not found"
        }
        '''
        
        result = subprocess.run(['powershell', '-Command', enable_cmd], capture_output=True, text=True)
        print(result.stdout)
        
        if "Stereo Mix enabled" in result.stdout:
            print("\nStereo Mix has been enabled successfully!")
            return True
        else:
            print("\nFailed to enable Stereo Mix.")
            print("Please check if your audio device supports Stereo Mix and enable it in Windows Sound settings:")
            print("1. Right-click the speaker icon in taskbar")
            print("2. Select 'Sound settings'")
            print("3. Click 'Sound Control Panel'")
            print("4. Go to 'Recording' tab")
            print("5. Right-click in the device list and check 'Show Disabled Devices'")
            print("6. Right-click 'Stereo Mix' and select 'Enable'")
            return False
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Attempting to enable Stereo Mix...")
    success = enable_stereo_mix(set_as_default=True)
    
    if success:
        print("\nStereo Mix configuration completed.")
        print("You can now use it in your recording applications.")
    else:
        print("\nFailed to configure Stereo Mix.")
        print("Please follow the manual instructions above.")
