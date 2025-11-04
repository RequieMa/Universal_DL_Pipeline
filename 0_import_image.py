import os
from label_studio_sdk.client import LabelStudio

# LABEL_STUDIO_URL = 'http://0.0.0.0:8080/'  # 替换为你的Label Studio地址
# API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODA1MTM5MTY4MywiaWF0IjoxNzQ0MTkxNjgzLCJqdGkiOiIzYjU5MDhiODM0NTQ0OWRhYjY0NWQ3NjFjY2QxMmQ1OSIsInVzZXJfaWQiOjF9.rwDpATpLySL5Qpor-ai-nIeSomQ5miTXqaQPLr6a2Bg'  # 从Label Studio的"Account & Settings"获取

# # 创建客户端实例
# ls = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=API_KEY)

image_folder = "anime_face"
# image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
# print(image_paths)

image_paths = [os.path.join(r'anime_face\000_hatsune_miku', f) for f in os.listdir(r'anime_face\000_hatsune_miku') if f.endswith(('.jpg', '.png', '.jpeg'))]
print(image_paths)

tasks = [
    {
        "image": f"/data/local-files/?d={os.path.relpath(path, start=r'anime_face\000_hatsune_miku')}",
        "label": "Anime"
    } 
    for path in image_paths
]
print(tasks)

# project_id = 1  # 替换为你的项目ID
# response = ls.projects.import_tasks(
#     id=project_id,
#     request=tasks,
#     preannotated_from_fields=['label']
# )
# print(f"成功导入 {len(tasks)} 张图片！")