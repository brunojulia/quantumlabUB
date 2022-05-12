# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(['principal.py'],
             pathex=[],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts, 
          [],
          exclude_binaries=True,
          name='principal',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas, 
               Tree('C:\\Users\\34626\\anaconda3\\envs\\clavsqua_env\\share\\sdl2\\bin'), 
               Tree('C:\\Users\\34626\\anaconda3\\envs\\clavsqua_env\\share\\glew\\bin'),
               Tree('C:\\Users\\34626\\anaconda3\\envs\\clavsqua_env\\Lib\\site-packages\\garden'),
               Tree('C:\\Users\\34626\\anaconda3\\envs\\clavsqua_env\\Lib\\site-packages\\kivy\garden'),
               Tree('C:\\Users\\34626\\OneDrive\\Escritorio\\6e SEMESTRE\\PRACS DEMPRESA\\eigengame'),
               Tree('C:\\Users\\34626\\OneDrive\\Escritorio\\6e SEMESTRE\\PRACS DEMPRESA\\eigengame\\graphs'),
               strip=False,
               upx=True,
               upx_exclude=[],
               name='principal')
