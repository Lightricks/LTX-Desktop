; Custom NSIS include for LTX Desktop installer
; Installs the VC++ 2015-2022 Redistributable (x64) required by PyTorch/CUDA

!macro customInstall
  ; Check if a sufficiently recent VC++ Redistributable is installed.
  ; Registry key covers the merged VC++ 2015-2022 family (major version 14).
  ; PyTorch compiled with MSVC 17.x needs at least build ~31000 (VS 2022 era).
  ReadRegDWORD $0 HKLM "SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\X64" "Installed"
  ReadRegDWORD $1 HKLM "SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\X64" "Bld"

  ${If} $0 != 1
  ${OrIf} $1 < 31000
    DetailPrint "Installing Visual C++ Redistributable..."
    File /oname=$PLUGINSDIR\vc_redist.x64.exe "${BUILD_RESOURCES_DIR}\vc_redist.x64.exe"
    ExecWait '"$PLUGINSDIR\vc_redist.x64.exe" /install /quiet /norestart' $0
    ${If} $0 != 0
      ; Non-zero exit — might be a reboot-required (3010) or actual error.
      ; Don't block installation; the app will still work in most cases and
      ; the user can install the redistributable manually if needed.
      DetailPrint "VC++ Redistributable installer exited with code $0"
    ${EndIf}
    Delete "$PLUGINSDIR\vc_redist.x64.exe"
  ${Else}
    DetailPrint "Visual C++ Redistributable already installed (build $1)."
  ${EndIf}
!macroend
