Unicode true
Target amd64-unicode
RequestExecutionLevel user
SetCompressor zlib
SetDatablockOptimize off
CRCCheck force
!include "MUI2.nsh"
!include "LogicLib.nsh"
!include "x64.nsh"
!include "Win\COM.nsh"
!define APP "LTX Desktop - AMD 395 MAX"
!define LEGACY_APP "LTX Desktop AMD"
!define DIAGNOSTIC_LINK "${APP} Diagnostics"
!define KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\LTXDesktopAMD"
Name "${APP}"
Caption "${APP} Setup"
UninstallCaption "${APP} Uninstall"
OutFile "${OUT}"
InstallDir "$LOCALAPPDATA\Programs\LTXDesktopAMD"
!define MUI_ICON "${ICONFILE}"
!define MUI_UNICON "${ICONFILE}"
VIProductVersion "1.3.3.1"
VIAddVersionKey "ProductName" "${APP}"
VIAddVersionKey "ProductVersion" "1.3.3.1"
VIAddVersionKey "FileDescription" "${APP} Setup"
VIAddVersionKey "FileVersion" "1.3.3.1"
VIAddVersionKey "LegalCopyright" "LTX Desktop contributors and component authors"
ShowInstDetails show
ShowUninstDetails show
!define MUI_ABORTWARNING
!define MUI_WELCOMEPAGE_TITLE "${APP}"
!ifdef OFFLINE
!define MUI_WELCOMEPAGE_TEXT "Install the Windows x64 edition for Radeon 8060S. Setup includes the application and native ROCm runtime. LTX downloads video model weights separately on first use.$\r$\n$\r$\nClose any running copy of ${APP} before starting."
!else
!ifdef COMPACT
!define MUI_WELCOMEPAGE_TEXT "Setup detects a supported NVIDIA CUDA GPU or Radeon 8060S and downloads its matching runtime. Other PCs use API mode. Model weights are available later from the app.$\r$\n$\r$\nClose LTX Desktop before starting."
!else
!define MUI_WELCOMEPAGE_TEXT "Install the Windows x64 edition with automatic NVIDIA CUDA and Radeon 8060S detection. Setup includes an app-managed ComfyUI renderer and downloads the matching runtime. LTX downloads video model weights separately on first use.$\r$\n$\r$\nClose any running copy of ${APP} before starting."
!endif
!endif
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_INSTFILES
!define MUI_FINISHPAGE_TEXT "Open ${APP} and choose Image to Video or Text to Video. Open the Model menu to set up and select the model you want. You can set up the included local renderer directly from its model panel.$\r$\n$\r$\nModel files download on demand."
!define MUI_FINISHPAGE_RUN "$INSTDIR\application\${APP}.exe"
!define MUI_FINISHPAGE_RUN_NOTCHECKED
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"
!macro DeleteOwnedShortcut LINK TARGET
  ${If} ${FileExists} "${LINK}"
    StrCpy $2 -1
    StrCpy $3 ""
    StrCpy $5 ""
    !insertmacro ComHlpr_CreateInProcInstance ${CLSID_ShellLink} ${IID_IShellLink} r0 ""
    ${If} $0 P<> 0
      ${IUnknown::QueryInterface} $0 '("${IID_IPersistFile}",.r1)'
      ${If} $1 P<> 0
        ${IPersistFile::Load} $1 '("${LINK}",${STGM_READ}).r2'
        ${If} $2 = 0
          ; SLGP_RAWPATH reads the stored destination even after an old launcher was removed.
          ${IShellLink::GetPath} $0 '(.r3,${NSIS_MAX_STRLEN},0,4).r2'
          ${If} $2 = 0
            ${IShellLink::GetArguments} $0 '(.r5,${NSIS_MAX_STRLEN}).r2'
          ${EndIf}
        ${EndIf}
        ${IUnknown::Release} $1 ""
      ${EndIf}
      ${IUnknown::Release} $0 ""
    ${EndIf}
    ${If} $2 = 0
    ${AndIf} $3 == "${TARGET}"
    ${AndIf} $5 == ""
      Delete "${LINK}"
    ${EndIf}
  ${EndIf}
!macroend

Section "${APP}"
  SetShellVarContext current
  SetRegView 64
  InitPluginsDir
  SetOutPath "$PLUGINSDIR\bundle"
  File "${SRC}/setup_engine.py"
  File "${SRC}/gpu_selection.py"
  File "${SRC}/cuda_runtime.py"
  File "${SRC}/cuda-runtime.json"
  File "${SRC}/diagnose.py"
  File "${SRC}/AMD-Diagnostics.cmd"
  File "${BUNDLE}/runtime-manifest.json"
  File "${BUNDLE}/sitecustomize.py"
  File "${BUNDLE}/repair_single_gpu.py"
  SetOutPath "$PLUGINSDIR\bundle\python"
  File /r "${BUNDLE}/python/*.*"
  SetOutPath "$PLUGINSDIR\bundle"
  SetCompress off
!ifdef OFFLINE
  File "${BUNDLE}/offline-index.json"
  File "${BUNDLE}/runtime-size.json"
!else
!ifdef COMPACT
  SetOutPath "$PLUGINSDIR\bundle\compact"
  File /r "${COMPACT}/*.*"
  SetOutPath "$PLUGINSDIR\bundle"
!else
  File "${BUNDLE}/online-index.json"
!endif
  File "${BUNDLE}/install_runtime.py"
  SetOutPath "$PLUGINSDIR\bundle\wheels"
  File /r "${BUNDLE}/wheels/*.*"
!endif
  SetOutPath "$PLUGINSDIR\bundle\vcredist"
  File /r "${BUNDLE}/vcredist/*.*"
  SetOutPath "$PLUGINSDIR\bundle"
  DetailPrint "Installing ${APP} and the selected GPU runtime..."
  nsExec::ExecToLog '"$PLUGINSDIR\bundle\python\python.exe" -u -X utf8 "$PLUGINSDIR\bundle\setup_engine.py" "$PLUGINSDIR\bundle" "$EXEDIR" "$EXEPATH"'
  Pop $0
  ${If} $0 != 0
    MessageBox MB_OK|MB_ICONSTOP "Setup stopped. See the installation details or open $LOCALAPPDATA\LTXDesktopAMD\setup\install.log."
    SetErrorLevel 1
    Abort
  ${EndIf}
  SetOutPath "$INSTDIR"
  WriteUninstaller "$INSTDIR\Uninstall.exe"
  ReadRegStr $4 HKCU "${KEY}" "DisplayName"
  ${If} $4 == "${LEGACY_APP}"
    !insertmacro DeleteOwnedShortcut "$SMPROGRAMS\${LEGACY_APP}\${LEGACY_APP}.lnk" "$INSTDIR\application\${LEGACY_APP}.exe"
    !insertmacro DeleteOwnedShortcut "$SMPROGRAMS\${LEGACY_APP}\Check Radeon GPU.lnk" "$INSTDIR\application\AMD-Diagnostics.cmd"
    !insertmacro DeleteOwnedShortcut "$DESKTOP\${LEGACY_APP}.lnk" "$INSTDIR\application\${LEGACY_APP}.exe"
    RMDir "$SMPROGRAMS\${LEGACY_APP}"
  ${EndIf}
  CreateDirectory "$SMPROGRAMS\${APP}"
  CreateShortcut "$SMPROGRAMS\${APP}\${APP}.lnk" "$INSTDIR\application\${APP}.exe"
  CreateShortcut "$SMPROGRAMS\${APP}\${DIAGNOSTIC_LINK}.lnk" "$INSTDIR\application\AMD-Diagnostics.cmd" "" "$INSTDIR\application\${APP}.exe"
  CreateShortcut "$DESKTOP\${APP}.lnk" "$INSTDIR\application\${APP}.exe"
  WriteRegStr HKCU "${KEY}" "DisplayName" "${APP}"
  WriteRegStr HKCU "${KEY}" "DisplayVersion" "1.3.3-community.1"
  WriteRegStr HKCU "${KEY}" "DisplayIcon" "$INSTDIR\application\${APP}.exe"
  WriteRegStr HKCU "${KEY}" "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteRegDWORD HKCU "${KEY}" "NoModify" 1
  WriteRegDWORD HKCU "${KEY}" "NoRepair" 1
SectionEnd
Section "Uninstall"
  SetShellVarContext current
  SetRegView 64
  IfFileExists "$INSTDIR\application\.ltx-amd-managed-v1" 0 missing
  Delete "$INSTDIR\application\${APP}.exe"
  IfFileExists "$INSTDIR\application\${APP}.exe" busy 0
  RMDir /r "$INSTDIR\application"
  IfFileExists "$INSTDIR\application\*.*" busy 0
  Goto cleanup
  busy:
    MessageBox MB_OK|MB_ICONSTOP "Close ${APP} and retry uninstall."
    SetErrorLevel 1
    Abort
  missing:
    IfFileExists "$INSTDIR\application\*.*" 0 cleanup
    MessageBox MB_OK|MB_ICONSTOP "The installed application's ownership marker is missing."
    SetErrorLevel 1
    Abort
  cleanup:
  !insertmacro DeleteOwnedShortcut "$SMPROGRAMS\${APP}\${DIAGNOSTIC_LINK}.lnk" "$INSTDIR\application\AMD-Diagnostics.cmd"
  !insertmacro DeleteOwnedShortcut "$SMPROGRAMS\${APP}\${APP}.lnk" "$INSTDIR\application\${APP}.exe"
  RMDir "$SMPROGRAMS\${APP}"
  !insertmacro DeleteOwnedShortcut "$DESKTOP\${APP}.lnk" "$INSTDIR\application\${APP}.exe"
  DeleteRegKey HKCU "${KEY}"
  Delete "$INSTDIR\Uninstall.exe"
  RMDir "$INSTDIR"
SectionEnd
