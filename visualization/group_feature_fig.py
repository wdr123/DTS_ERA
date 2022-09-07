import os
import shutil



heatmap_path = '../results/visualization/results_heatmap'
heatmap_folder = os.listdir(heatmap_path)
heatmap_des = '../results/visualization/group'


if not os.path.exists(os.path.join(heatmap_des, 'left_gaze')):
    os.makedirs(os.path.join(heatmap_des, 'left_gaze'))
if not os.path.exists(os.path.join(heatmap_des, 'right_gaze')):
    os.makedirs(os.path.join(heatmap_des, 'right_gaze'))
if not os.path.exists(os.path.join(heatmap_des, 'touch_visualization')):
    os.makedirs(os.path.join(heatmap_des, 'touch_visualization'))
if not os.path.exists(os.path.join(heatmap_des, 'touch_hardness')):
    os.makedirs(os.path.join(heatmap_des, 'touch_hardness'))


for user_level_folder in heatmap_folder:
    attn_fig_folder_path = os.path.join(heatmap_path, user_level_folder)
    for attn_fig in os.listdir(attn_fig_folder_path):
        fig_path = os.path.join(attn_fig_folder_path, attn_fig)
        if 'left_gaze' in attn_fig:
            if not os.path.exists(os.path.join(heatmap_des+'/left_gaze', fig_path.split('\\')[-2])):
                os.makedirs(os.path.join(heatmap_des+'/left_gaze', fig_path.split('\\')[-2]))
            if 'origin' not in fig_path:
                shutil.copy(fig_path, os.path.join(heatmap_des+'/left_gaze'+'/'+fig_path.split('\\')[-2], attn_fig))
        elif 'right_gaze' in attn_fig:
            if not os.path.exists(os.path.join(heatmap_des+'/right_gaze', fig_path.split('\\')[-2])):
                os.makedirs(os.path.join(heatmap_des+'/right_gaze', fig_path.split('\\')[-2]))
            if 'origin' not in fig_path:
                shutil.copy(fig_path, os.path.join(heatmap_des+'/right_gaze'+'/'+fig_path.split('\\')[-2], attn_fig))
        elif 'touch_visualization' in attn_fig:
            if not os.path.exists(os.path.join(heatmap_des+'/touch_visualization', fig_path.split('\\')[-2])):
                os.makedirs(os.path.join(heatmap_des+'/touch_visualization', fig_path.split('\\')[-2]))
            if 'origin' not in fig_path:
                shutil.copy(fig_path, os.path.join(heatmap_des+'/touch_visualization'+'/'+fig_path.split('\\')[-2], attn_fig))
        else:
            if not os.path.exists(os.path.join(heatmap_des+'/touch_hardness', fig_path.split('\\')[-2])):
                os.makedirs(os.path.join(heatmap_des+'/touch_hardness', fig_path.split('\\')[-2]))
            if 'origin' not in fig_path:
                shutil.copy(fig_path, os.path.join(heatmap_des+'/touch_hardness'+'/'+fig_path.split('\\')[-2], attn_fig))