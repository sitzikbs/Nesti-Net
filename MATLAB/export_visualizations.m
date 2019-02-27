clear all
close all
clc

data_path = '/home/itzik/PycharmProjects/NestiNet/data/pcpnet/';
results_path = '/home/itzik/PycharmProjects/NestiNet/log/experts/pcpnet_results/';
file_sets_path = '/home/itzik/PycharmProjects/NestiNet/data/pcpnet/file_sets/'; % directory containing lists of files ,camera parameters and shape list
output_path = [results_path, 'images/'] ;
file_list_file_name = 'testset_all.txt';
shape_list_file_name = 'shapes_list_test.txt';

export_type = 'all'; %shape name for single point cloud export or all for all point clouds

export_point_vis = true;
export_normal_vis = true;
export_expert_vis = true;
export_error_vis = true;
% export_curvature_vis = true;
n_experts = 7;
max_err_ang = 60;

use_subset = false;
use_hpr = false;

if use_subset
    point_size = 400;
else
     point_size = 10;
end

if ~exist(output_path, 'dir')
    mkdir(output_path)
end

% directory for exporting XYZ vis
if export_point_vis
    xyz_img_output_path = [output_path, 'xyz/'];
    if ~exist(xyz_img_output_path,'dir')
        mkdir(xyz_img_output_path)
    end
end 
% directory for exporting normal vis
if export_normal_vis
    normals_img_output_path = [output_path, 'normals/'];
    if ~exist(normals_img_output_path,'dir')
        mkdir(normals_img_output_path)
    end
end 
% directory for exporting experts vis
if export_expert_vis
    expert_img_output_path = [output_path, 'experts/'];
    if ~exist(expert_img_output_path, 'dir')
        mkdir(expert_img_output_path)
    end
    % export expert color map
        fig_h = figure('color','w');
        n_colors = n_experts;
        colors = distinguishable_colors(n_colors);
        expert_legend(n_experts, colors, 'horizontal');
        axis off
        image_filename = [expert_img_output_path, 'color_map.png'];    
        print(image_filename, '-dpng')
        close(fig_h);
end 
% directory for exporting experts vis
if export_error_vis
   error_img_output_path = [output_path, 'errors/'];
    if ~exist(error_img_output_path, 'dir')
        mkdir(error_img_output_path)
    end
        fig_h = figure('color','w');
        colormap('parula');
%         caxis([0, max_err_ang]);
        colorbar('location', 'south', 'Ticks',[0, 1], 'TickLabels',[0, max_err_ang]);
        axis off
        image_filename = [error_img_output_path, 'color_bar.png'];    
        print(image_filename, '-dpng')
        close(fig_h);
end 

% if export_curvature_vis
%     curvatures_img_output_path = [output_path, 'curvatures/'];
%     if ~exist(curvatures_img_output_path,'dir')
%         mkdir(curvatures_img_output_path)
%     end
% end 

xyz_file_list = dir([data_path,'*.xyz']);
cam_params_file =  [file_sets_path, 'camera_parameters.txt'];
shapes_list = [file_sets_path, shape_list_file_name];
file_list_to_export = [file_sets_path, file_list_file_name];
shapes_list = strsplit(fileread(shapes_list));
shapes_list = shapes_list(~cellfun('isempty',shapes_list));  % remove empty cells
shapes_to_export = strsplit(fileread(file_list_to_export));
shapes_to_export = shapes_to_export(~cellfun('isempty',shapes_to_export));  % remove empty cells
scale_list = zeros(size(shapes_list));
mean_list = zeros(size(shapes_list,2), 3);
if ~strcmp(export_type, 'all')
    shapes_to_export = shapes_to_export(contains( shapes_to_export, export_type));
end
cam_params = dlmread(cam_params_file);

for shape = shapes_to_export
    disp(['saving ', shape{1}, '...']);
    xyz_file_name = [data_path, shape{1}, '.xyz'];
    normals_gt_file_name = [data_path, shape{1}, '.normals'];
    normals_file_name =  [results_path, shape{1}, '.normals'];
    expert_file_name = [results_path, shape{1}, '.experts'];
    idx_file_name =  [data_path, shape{1}, '.pidx'];
    curvatures_gt_file_name = [data_path, shape{1}, '.curv'];
    curvatures_file_name = [results_path, shape{1}, '.curv'];
    points = dlmread(xyz_file_name);
    
    for i= 1:size(shapes_list, 2) 
        if contains(shape, shapes_list{i})
            shape_idx = i;
        end
    end
    
    if scale_list(shape_idx) == 0 
         mean_list(shape_idx, :) = mean(points);
         points = points - mean_list(shape_idx, :); 
         scale_list(shape_idx) = (1./max(sqrt(sum(points.^2, 2))));
         points = points.*scale_list(shape_idx); 
    else
         points = points - mean_list(shape_idx, :); 
         points = points.*scale_list(shape_idx); 
    end
    normals_gt = dlmread(normals_gt_file_name);
    normals = dlmread(normals_file_name);
    n_normals = size(normals, 1);
    npoints = size(points,1);

    if npoints ~= n_normals
         idxs = dlmread(idx_file_name) + 1;
         points = points(idxs, :);
         normals_gt = normals_gt(idxs, :);
    elseif use_subset
        idxs = dlmread(idx_file_name) + 1;
        points = points(idxs, :);
        normals_gt = normals_gt(idxs, :);
        normals = normals(idxs, :);
    end
   npoints = size(points,1); % new number of points
    visiblePtInds = 1:npoints;
    
    fig_h = figure();
    ax_h = axes('position',[0, 0, 1, 1]);
    set_vis_props(fig_h, ax_h);
    if ~strcmp(shapes_list{shape_idx}, 'Cup34100k')
     	Reshape(gca, cam_params(shape_idx, 1), cam_params(shape_idx, 2), cam_params(shape_idx, 3));
    else
        Reshape_cup(gca, cam_params(shape_idx, 1), cam_params(shape_idx, 2), cam_params(shape_idx, 3));
    end
%     Reshape(gca,view_rad, view_phi, view_theta);
    xlim([-1, 1]);
    ylim([-1, 1]);
    zlim([-1, 1]);
    if use_hpr
        visiblePtInds = HPR(points, ax_h.CameraPosition, 3);
    end
    points = points (visiblePtInds,:);
    normals_gt = normals_gt(visiblePtInds,:);
    normals = normals(visiblePtInds,:);
    
    pc_h = scatter3(points(:, 1), points(:, 2), points(:, 3), point_size, '.');
    axis off
    if export_point_vis
        image_filename = [xyz_img_output_path, shape{1}, '.png'];
        print(image_filename, '-dpng')
    end 
    
    if export_normal_vis
        mapped_normal_colors = Sphere2RGBCube(sign(sum(normals_gt.*normals,2)).*normals);
        pc_h.CData = mapped_normal_colors;
        image_filename = [normals_img_output_path, shape{1}, '.png'];
        print(image_filename, '-dpng')
    end
    
    if export_error_vis     
%         error = min(sqrt(sum((normals_gt - normals).^2, 2)), sqrt(sum((normals_gt + normals).^2, 2))); % consider visualizing the sin/ cosine error
        diff = abs(sum(normals.*normals_gt,2))./ (sqrt(sum(normals.^2,2)).* sqrt(sum(normals_gt.^2,2)));
        diff(diff > 1) = 1;
        error = acosd(diff);
        rms = mean(error);
        colormap('parula');
        caxis([0, max_err_ang]);
        pc_h.CData = error;
        active_axes = gca;
        active_axes.Position =  [0, 0, 1, 0.9];
        ax_rns_text = axes('position',[0, 0.9, 1, 0.1]);
        axis off
        text('string',num2str(rms), 'position',[0.5, 0.5], 'FontSize', 24, 'HorizontalAlignment', 'center')
        image_filename = [error_img_output_path, shape{1}, '.png'];    
        print(image_filename, '-dpng')
        gcf.CurrentAxes  = active_axes;
        active_axes.Position =  [0, 0, 1, 1];
        delete(ax_rns_text)
    end
    
    if export_expert_vis
        expert = dlmread(expert_file_name) + 1;
%         expert = expert(idxs, :);
%         expert = expert(visiblePtInds,:);
        n_colors = n_experts;
        colors = distinguishable_colors(n_colors);
%         active_axes = gca;
%         active_axes.Position =  [0, 0, 0.8, 1];
%         axes('position',[0.8, 0, 0.2, 1]);
%         axis off
%         expert_legend(n_experts, colors, 'vertical');
%         gcf.CurrentAxes  = active_axes;
        colormap(colors);
        pc_h.CData =colors( expert, :);
        image_filename = [expert_img_output_path, shape{1}, '.png'];    
        print(image_filename, '-dpng')
    end
    
%     if export_curvature_vis
%         curvatures_gt = dlmread(curvatures_gt_file_name);
%         gaussian_curvature_gt = curvatures_gt(:,1) .* curvatures_gt(:, 2);
%          pc_h.CData = gaussian_curvature_gt;
%         image_filename = [curvature_img_output_path, shape{1}, '.png'];    
%         print(image_filename, '-dpng')
%          
%     end

    close(fig_h);
end
disp('All done!');