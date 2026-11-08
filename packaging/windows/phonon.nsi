; Phonon for Windows installer (NSIS 3). Built by scripts/package-win-installer.sh, which passes
;   /DVERSION=x.y.z /DEXE=<phonon-win.exe> /DVCREDIST=<vc_redist.x64.exe> /DOUT=<installer.exe>
; Per-user install (no admin) to %LOCALAPPDATA%\Programs\Phonon. The speech and correction runtimes that Phonon
; downloads on first run link the Microsoft Visual C++ runtime, which a clean Windows 11 does not have; the installer
; carries Microsoft's redistributable and runs it (one UAC prompt) only when the x64 runtime is not registered.

Unicode true
SetCompressor /SOLID lzma
RequestExecutionLevel user
Name "Phonon"
OutFile "${OUT}"
InstallDir "$LOCALAPPDATA\Programs\Phonon"
InstallDirRegKey HKCU "Software\Phonon" "InstallDir"
BrandingText "Phonon ${VERSION}"

!include "MUI2.nsh"
!include "LogicLib.nsh"
!include "x64.nsh"

; makensis runs from this script's folder
!define MUI_ICON "..\..\assets\phonon.ico"
!define MUI_UNICON "..\..\assets\phonon.ico"
!define UNINST_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\Phonon"
!define MUI_FINISHPAGE_RUN "$INSTDIR\phonon-win.exe"
!define MUI_FINISHPAGE_RUN_TEXT "Start Phonon now (hold Right Ctrl to dictate)"
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

VIProductVersion "${VERSION}.0"
VIAddVersionKey "ProductName" "Phonon"
VIAddVersionKey "FileDescription" "Phonon installer"
VIAddVersionKey "FileVersion" "${VERSION}"
VIAddVersionKey "ProductVersion" "${VERSION}"
VIAddVersionKey "LegalCopyright" "GPL-3.0"

Function .onInit
  ${IfNot} ${RunningX64}
    MessageBox MB_ICONSTOP "Phonon needs 64-bit Windows 10 or 11."
    Abort
  ${EndIf}
FunctionEnd

Section "Phonon" SecMain
  ; a running copy holds its executable open, and taskkill returns before the process is gone: wait until the old
  ; file can be deleted (up to 10 s), or a silent upgrade quietly keeps the old executable
  nsExec::Exec 'taskkill /IM phonon-win.exe /F'
  StrCpy $1 0
  ${Do}
    ClearErrors
    Delete "$INSTDIR\phonon-win.exe"
    ${IfNot} ${FileExists} "$INSTDIR\phonon-win.exe"
      ${Break}
    ${EndIf}
    IntOp $1 $1 + 1
    ${If} $1 > 40
      ${Break}
    ${EndIf}
    Sleep 250
  ${Loop}
  SetOutPath "$INSTDIR"
  SetOverwrite on
  File "/oname=phonon-win.exe" "${EXE}"

  ; Visual C++ runtime (x64), only when missing
  SetRegView 64
  ReadRegDWORD $0 HKLM "SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" "Installed"
  SetRegView 32
  ${If} $0 != 1
    DetailPrint "Installing the Microsoft Visual C++ runtime..."
    InitPluginsDir
    File "/oname=$PLUGINSDIR\vc_redist.x64.exe" "${VCREDIST}"
    ExecShellWait "runas" "$PLUGINSDIR\vc_redist.x64.exe" "/install /quiet /norestart"
    SetRegView 64
    ReadRegDWORD $0 HKLM "SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" "Installed"
    SetRegView 32
    ${If} $0 != 1
      MessageBox MB_ICONEXCLAMATION "The Microsoft Visual C++ runtime was not installed (the permission prompt may have been declined). Phonon will tell you how to install it when it starts." /SD IDOK
    ${EndIf}
  ${EndIf}

  CreateShortcut "$SMPROGRAMS\Phonon.lnk" "$INSTDIR\phonon-win.exe"
  ; start at sign-in, like the macOS app
  CreateShortcut "$SMSTARTUP\Phonon.lnk" "$INSTDIR\phonon-win.exe"

  WriteUninstaller "$INSTDIR\Uninstall Phonon.exe"
  WriteRegStr HKCU "Software\Phonon" "InstallDir" "$INSTDIR"
  WriteRegStr HKCU "${UNINST_KEY}" "DisplayName" "Phonon"
  WriteRegStr HKCU "${UNINST_KEY}" "DisplayVersion" "${VERSION}"
  WriteRegStr HKCU "${UNINST_KEY}" "Publisher" "Phonon"
  WriteRegStr HKCU "${UNINST_KEY}" "DisplayIcon" "$INSTDIR\phonon-win.exe"
  WriteRegStr HKCU "${UNINST_KEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "${UNINST_KEY}" "URLInfoAbout" "https://github.com/Infatoshi/phonon"
  WriteRegStr HKCU "${UNINST_KEY}" "UninstallString" '"$INSTDIR\Uninstall Phonon.exe"'
  WriteRegStr HKCU "${UNINST_KEY}" "QuietUninstallString" '"$INSTDIR\Uninstall Phonon.exe" /S'
  WriteRegDWORD HKCU "${UNINST_KEY}" "NoModify" 1
  WriteRegDWORD HKCU "${UNINST_KEY}" "NoRepair" 1
  WriteRegDWORD HKCU "${UNINST_KEY}" "EstimatedSize" 1700000
SectionEnd

Section "Uninstall"
  nsExec::Exec 'taskkill /IM phonon-win.exe /F'
  nsExec::Exec 'taskkill /IM llama-server.exe /F'
  Sleep 500
  Delete "$INSTDIR\phonon-win.exe"
  Delete "$INSTDIR\Uninstall Phonon.exe"
  RMDir "$INSTDIR"
  Delete "$SMPROGRAMS\Phonon.lnk"
  Delete "$SMSTARTUP\Phonon.lnk"
  DeleteRegKey HKCU "${UNINST_KEY}"
  DeleteRegKey HKCU "Software\Phonon"
  ; the speech and correction models (about 1.6 GB) and recordings live in %LOCALAPPDATA%\Phonon
  ${If} ${FileExists} "$LOCALAPPDATA\Phonon\*.*"
    MessageBox MB_YESNO|MB_ICONQUESTION "Also delete Phonon's downloaded models and recordings ($LOCALAPPDATA\Phonon)?" /SD IDYES IDNO keep
    RMDir /r "$LOCALAPPDATA\Phonon"
    keep:
  ${EndIf}
SectionEnd
