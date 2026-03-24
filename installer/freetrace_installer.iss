; Made by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15
; Inno Setup Script for FreeTrace C++
; Download Inno Setup: https://jrsoftware.org/isinfo.php
;
; Build with: iscc freetrace_installer.iss
; Or open in Inno Setup Compiler GUI and click Build.
;
; Before building, run build_installer.bat to prepare the staging directory.
;
; The installer is FULLY SELF-CONTAINED.
; The user only needs an NVIDIA GPU driver installed — everything else is bundled
; (CUDA runtime, cuDNN, ONNX Runtime, libtiff, libpng, VC++ runtime).

#define MyAppName "FreeTrace"
#define MyAppVersion "1.6.1.2"
#define MyAppPublisher "Junwoo PARK"
#define MyAppURL "https://github.com/JunwooParkSaribu/FreeTrace_cpp"
#define MyAppExeName "FreeTrace_GUI.exe"
#define StagingDir "..\installer_staging"

[Setup]
AppId={{E7A3F1B2-4C5D-6E7F-8A9B-0C1D2E3F4A5B}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
; Output installer to installer/ directory
OutputDir=.
OutputBaseFilename=FreeTrace_{#MyAppVersion}_win64_setup
; Require 64-bit Windows
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; Compression
Compression=lzma2/fast
SolidCompression=yes
; Appearance
WizardStyle=modern
SetupIconFile={#StagingDir}\icon\freetrace_icon.ico
LicenseFile=
; Privileges — install per-user by default (no admin needed)
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "addtopath"; Description: "Add FreeTrace to system PATH"; GroupDescription: "System integration:"; Flags: unchecked

[Files]
; === Main executables ===
Source: "{#StagingDir}\freetrace.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#StagingDir}\FreeTrace_GUI.exe"; DestDir: "{app}"; Flags: ignoreversion
; === GUI Python runtime (one-folder mode for fast startup) ===
Source: "{#StagingDir}\_internal\*"; DestDir: "{app}\_internal"; Flags: ignoreversion recursesubdirs createallsubdirs

; === Model files ===
Source: "{#StagingDir}\models\*"; DestDir: "{app}\models"; Flags: ignoreversion recursesubdirs

; === Icon ===
Source: "{#StagingDir}\icon\*"; DestDir: "{app}\icon"; Flags: ignoreversion recursesubdirs skipifsourcedoesntexist

; === ONNX Runtime DLLs ===
Source: "{#StagingDir}\onnxruntime.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#StagingDir}\onnxruntime_providers_cuda.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#StagingDir}\onnxruntime_providers_shared.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist

; === CUDA 12.x runtime DLLs ===
Source: "{#StagingDir}\cudart64_12.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#StagingDir}\cublas64_12.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#StagingDir}\cublasLt64_12.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#StagingDir}\cufft64_11.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#StagingDir}\curand64_10.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#StagingDir}\cusolver64_11.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#StagingDir}\cusparse64_12.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#StagingDir}\nvJitLink_120_0.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#StagingDir}\nvrtc64_120_0.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist

; === cuDNN 9.x DLLs ===
Source: "{#StagingDir}\cudnn64_9.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#StagingDir}\cudnn_cnn64_9.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#StagingDir}\cudnn_ops64_9.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#StagingDir}\cudnn_graph64_9.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#StagingDir}\cudnn_engines_precompiled64_9.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#StagingDir}\cudnn_engines_runtime_compiled64_9.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#StagingDir}\cudnn_heuristic64_9.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#StagingDir}\cudnn_adv64_9.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist

; === libtiff + libpng + zlib (from vcpkg) ===
Source: "{#StagingDir}\tiff.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#StagingDir}\tiffxx.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#StagingDir}\libpng16.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#StagingDir}\png16.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#StagingDir}\zlib1.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#StagingDir}\jpeg62.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#StagingDir}\turbojpeg.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#StagingDir}\liblzma.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#StagingDir}\libwebp.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#StagingDir}\libsharpyuv.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#StagingDir}\Lerc.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#StagingDir}\libdeflate.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#StagingDir}\zstd.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist

; === zlib for cuDNN ===
Source: "{#StagingDir}\zlibwapi.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist

; === Visual C++ Runtime ===
Source: "{#StagingDir}\vcruntime140.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#StagingDir}\vcruntime140_1.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#StagingDir}\msvcp140.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#StagingDir}\concrt140.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; IconFilename: "{app}\icon\freetrace_icon.ico"
Name: "{group}\{#MyAppName} (CLI)"; Filename: "cmd.exe"; Parameters: "/k ""{app}\freetrace.exe"" --help"; WorkingDir: "{app}"; IconFilename: "{app}\icon\freetrace_icon.ico"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon; IconFilename: "{app}\icon\freetrace_icon.ico"

[Registry]
; Add to PATH if user selected that task
Root: HKCU; Subkey: "Environment"; ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app}"; Tasks: addtopath; Check: NeedsAddPath('{app}')

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent

[Code]
function NeedsAddPath(Param: string): Boolean;
var
  OrigPath: string;
begin
  if not RegQueryStringValue(HKEY_CURRENT_USER,
    'Environment', 'Path', OrigPath)
  then begin
    Result := True;
    exit;
  end;
  Result := Pos(';' + Param + ';', ';' + OrigPath + ';') = 0;
end;
