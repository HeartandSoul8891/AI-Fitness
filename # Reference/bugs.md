after saving in settings:

TypeError: save_settings() missing 1 required positional argument: 'settings_dict'
Traceback:
File "D:\VSCode\AI-Fitness\main.py", line 44, in <module>
    main()
File "D:\VSCode\AI-Fitness\main.py", line 40, in main
    settings()
File "D:\VSCode\AI-Fitness\tabs\settings_tab.py", line 121, in render_ui
    save_settings()

auto-Tagger BBOX ->  SEGM

select yolo model -> "custom model path" -> remove
                  -> chose yolo model file -> broken most likely cos settings is broken


doinga test run gives error:

TypeError: run_auto_tagger() got an unexpected keyword argument 'mask_shape'
Traceback:
File "D:\VSCode\AI-Fitness\main.py", line 44, in <module>
    main()
File "D:\VSCode\AI-Fitness\main.py", line 32, in main
    ultralytics()
File "D:\VSCode\AI-Fitness\tabs\ultralytics\ultralytics_tab.py", line 42, in main
    tsegm()       # Renders auto_tagger_segm_tab() (keys: segm_tagger_*)
    ^^^^^^^
File "D:\VSCode\AI-Fitness\tabs\ultralytics\auto_tagger_bbox_segm_tab.py", line 211, in auto_tagger_segm_tab
    result = run_auto_tagger(
             ^^^^^^^^^^^^^^^^

