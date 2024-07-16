/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

/**
 * Original Projection
 * -------------------------------------------------------------------------------------------------------------------------------------
 * Used only to determine the tiles involved in the Gaussian ellipsoids; other methods can optionally be used for this determination.
 * Compared to the bounding box, the correctness of the function values is more important. (for panorama camera only)
*/
// Forward version of 2D covariance matrix computation 
__device__ float3 computeNaiveCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	/**
	 * ------------------------------- For Optimal GS (panorama) ------------------------------------------
	*/
	// const float limx = 1.3f * tan_fovx;
	// const float limy = 1.3f * tan_fovy;
	// const float txtz = t.x / t.z;
	// const float tytz = t.y / t.z;
	// t.x = min(limx, max(-limx, txtz)) * t.z;
	// t.y = min(limy, max(-limy, tytz)) * t.z;

	// glm::mat3 J = glm::mat3(
	// 	focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
	// 	0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
	// 	0, 0, 0);

	float pi = 3.1415926f;
	float x2 = t.x * t.x;
	float y2 = t.y * t.y;
	float z2 = t.z * t.z;
	float x2y2z2 = x2 + y2 + z2;

	// if (x2y2z2 < 0.04f) {
	// 	printf("[Log] in forward.cu computeCov2D x2y2z2=%f", x2y2z2);
	// }
	// glm::mat3 J = glm::mat3(
	// 	focal_x / (2.0f * pi) * (-1.0f / t.z) / (1 + (t.x / t.z) * (t.x / t.z)), 0.0f, focal_x / (2.0f * pi) * (t.x / (t.z * t.z)) / (1 + (t.x / t.z) * (t.x / t.z)),
	// 	-1.0f * focal_y * t.x * t.y / (pi * sqrt(x2 + z2 * x2y2z2)), focal_y * sqrt(x2 + z2) / (pi * x2y2z2), -1.0 * focal_y * t.y * t.z / (pi * sqrt(x2 + z2 * x2y2z2)),
	// 	0, 0, 0);


	// precompute    approximate
	// glm::mat3 J = glm::mat3(
	// 	focal_x / (2.0f * pi) * (t.z) / (x2 + z2), 0.0f, -focal_x / (2.0f * pi) * (t.x) / (x2 + z2),
	// 	-1.0f * focal_y * t.x * t.y / (pi * sqrt(x2 + z2 * x2y2z2)), focal_y * sqrt(x2 + z2) / (pi * x2y2z2), -1.0 * focal_y * t.y * t.z / (pi * sqrt(x2 + z2 * x2y2z2)),
	// 	0, 0, 0);
	
	float coef_x = (focal_x / (2.0f * pi)) ;
	float coef_y = (focal_y / (pi));
	glm::mat3 J = glm::mat3(
		coef_x * (t.z) / (x2 + z2), 0.0f, coef_x * -(t.x) / (x2 + z2),
		coef_y * -t.x * t.y / (sqrt(x2 + z2 * x2y2z2)), coef_y * sqrt(x2 + z2) / (x2y2z2), coef_y * -t.y * t.z / (sqrt(x2 + z2 * x2y2z2)),
		0, 0, 0); 


	bool DEBUG = false;

	if (DEBUG) 
	{
		printf("[Log] forward.cu computeCov2D t: (%f, %f, %f), focal_x: %f, focal_y: %f\n", t.x, t.y, t.z, focal_x, focal_y);
	}
	/**
	 * ------------------------------- For Optimal GS (panorama) ------------------------------------------
	*/


	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += KERNEL_SIZE;
	cov[1][1] += KERNEL_SIZE;
	
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

/**
 * Optimal Projection (When adapting to other cameras, no changes for codes are needed here, so the local affine approximation error will not be affected by the camera model.)
*/
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 15
	// and 18 in "On the Error Analysis of 3D Gaussian Splatting and an Optimal Projection Strategy" (Huang et al., 2024). 
	float3 t = transformPoint4x3(mean, viewmatrix);

	/**
	 * ------------------------------- For Optimal GS (panorama) ------------------------------------------
	*/
	const float fx = 512.f;
	const float fy = 512.f;
	// Project the Gaussian's mean onto the tangent plane
	float dis_inv = 1.0f / (sqrt(t.x * t.x + t.y * t.y + t.z * t.z) + 0.0000001f);
	float3 mu = { t.x * dis_inv, t.y * dis_inv, t.z * dis_inv};

	// Convert the global ray coordinate to the local tangent plane coordinate. (Equation 18 i.e. invertible matrix Q in paper)
	float mut_xyz = mu.x * t.x + mu.y * t.y + mu.z * t.z;
	float mut_xyz2 = mut_xyz * mut_xyz;
	float mut_xyz2_inv = 1.0f / (mut_xyz2 + 0.0000001f);

	// Cartesian to polar coordinates.
	float theta = atan2(-mu.y, sqrt(mu.x * mu.x + mu.z * mu.z));
	float phi = atan2(mu.x, mu.z);

	// To reduce the number of sine and cosine function calculations.
	float sin_phi = sin(phi);
	float cos_phi = cos(phi);

	float sin_theta = sin(theta);
	float cos_theta = cos(theta);

	// (Equation 15 and 18 in paper i.e. Q * J)
	glm::mat3 J = glm::mat3(
		fx * (
			(mu.x * t.z * sin_phi + mu.y * t.y * cos_phi + mu.z * t.z * cos_phi) * mut_xyz2_inv
		),
		fx * (
			(mu.y * (-t.x * cos_phi + t.z * sin_phi)) * mut_xyz2_inv
		),
		fx * (
			-(mu.x * t.x * sin_phi + mu.y * t.y * sin_phi + mu.z * t.x * cos_phi) * mut_xyz2_inv
		),

		fy * (
			(-mu.x * t.y * cos_theta - mu.x * t.z * sin_theta * cos_phi + mu.y * t.y * sin_phi * sin_theta + mu.z * t.z * sin_phi * sin_theta) * mut_xyz2_inv
		),
		fy * (
			(mu.x * t.x * cos_theta - mu.y * t.x * sin_phi * sin_theta - mu.y * t.z * sin_theta * cos_phi + mu.z * t.z * cos_theta) * mut_xyz2_inv
		),
		fy * (
			(mu.x * t.x * sin_theta * cos_phi + mu.y * t.y * sin_theta * cos_phi - mu.z * t.x * sin_phi * sin_theta - mu.z * t.y * cos_theta) * mut_xyz2_inv
		),
		0.0f,
		0.0f,
		0.0f
	);
	
	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += KERNEL_SIZE;
	cov[1][1] += KERNEL_SIZE;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}


// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
/**
 * ------------------------------- For Optimal GS (panorama) ------------------------------------------
*/
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	/**
	 * ------------------------------- For Optimal GS (panorama) ------------------------------------------
	*/
	// float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	// float p_w = 1.0f / (p_hom.w + 0.0000001f);
	// float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	float4 p_hom = transformPoint4x4(p_orig, viewmatrix);
	float p_w = 1.0f / (sqrt(p_hom.x * p_hom.x + p_hom.y * p_hom.y + p_hom.z * p_hom.z) + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w};
	/**
 	 * ------------------------------- For Optimal GS (panorama) ------------------------------------------
	*/

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	
	float3 naive_cov = computeNaiveCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	float mid = 0.5f * (naive_cov.x + naive_cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	
	// printf("my_radius: %f\n", my_radius);
	/**
	 * ------------------------------- For Optimal GS (panorama) ------------------------------------------
	*/
	float theta = atan2(-p_proj.y, sqrt(p_proj.x * p_proj.x + p_proj.z * p_proj.z)); // asin(-p_orig.y);
	float phi = atan2(p_proj.x, p_proj.z);
	float pi = 3.1415926f;

	phi /= pi;
	theta /= -(pi / 2);
	// float2 point_image = { 1.0 * W * phi / 2 + 1.0 * W / 2, -1.0 * theta * H / 2 + 1.0 * H / 2};
	float2 point_image = { 
		ndc2Pix(phi, W), 
		ndc2Pix(theta, H) 
	};
	
	// bool DEBUG = false;

	// if (DEBUG) 
	// {
	// 	float x = cos(theta) * sin(phi);
	// 	float y = -sin(theta);
	// 	float z = cos(theta) * cos(phi);
	// 	printf("[Log] p_proj: (%f, %f, %f), xyz: (%f, %f, %f), uv: (%f, %f)\n", 
	// 	p_proj.x, p_proj.y, p_proj.z, 
	// 	x, y, z,
	// 	point_image.x, point_image.y);
	// }
	/**
	 * ------------------------------- For Optimal GS (panorama) ------------------------------------------
	*/

	uint2 rect_min, rect_max;

	// //******************************************************************
	// my_radius = 64;

	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	// The distance between two points should be more suitable for depth sorting compared to the projection of the distance along the z-axis.
	depths[idx] = sqrt(p_view.x * p_view.x + p_view.y * p_view.y + p_view.z * p_view.z);

	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

	const float pi = 3.1415926f;
	const float pi_invh = pi / H;
	const float pi_invw = pi / W;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{

		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j]; // panorama mean2D
			// pixf panorama (uv)

			// xy => sphere(mu_x, mu_y, mu_z) = (mu_x, mu_y, mu_z) => z_plane(new_mu_x, new_mu_y, new_mu_z)
			// uv => sphere(x, y, z) = (mu_x, mu_y, mu_z) => z_plane(new_x, new_y, new_z)

			float theta_xy = (H * 0.5f - xy.y) * pi_invh; 
			float phi_xy = (xy.x - W * 0.5f) * 2.0f * pi_invw;

			float theta_pixf = (H * 0.5f - pixf.y) * pi_invh;
			float phi_pixf = (pixf.x - W * 0.5f) * 2.0f * pi_invw;

			const float fx = 512.f;
			const float fy = 512.f;

			float sin_phi = sin(phi_xy);
			float cos_phi = cos(phi_xy);

			float sin_theta = sin(theta_xy);
			float cos_theta = cos(theta_xy);

			float3 mu = {
				cos(theta_xy) * sin(phi_xy),
				-sin(theta_xy),
				cos(theta_xy) * cos(phi_xy)
			};

			float3 t = {
				cos(theta_pixf) * sin(phi_pixf),
				-sin(theta_pixf),
				cos(theta_pixf) * cos(phi_pixf)
			};

			if (mu.x * t.x + mu.y * t.y + mu.z * t.z < 0.0000001f)  // 0.0000001f
			{
				continue;
			}

			// float u_xy = fx * (mu.x * mu.x + mu.y * mu.y + mu.z * mu.x);
			// float v_xy = fy * (mu.x * mu.x + mu.y * mu.y + mu.z * mu.y);

			// float u_pixf = fx * (mu.x * t.x + mu.y * t.y + mu.z * t.x) / (mu.x * t.x + mu.y * t.y + mu.z * t.z + 0.0000001f);
			// float v_pixf = fy * (mu.x * t.x + mu.y * t.y + mu.z * t.y) / (mu.x * t.x + mu.y * t.y + mu.z * t.z + 0.0000001f);


			float u_xy = 0.0f;
			float v_xy = 0.0f;

			float uv_pixf_inv = 1.f / (mu.x * t.x + mu.y * t.y + mu.z * t.z);

			float u_pixf = fx * (t.x * cos_phi - t.z * sin_phi) * uv_pixf_inv;
			float v_pixf = fy * (t.x * sin_phi * sin_theta + t.y * cos_theta + t.z * sin_theta * cos_phi) * uv_pixf_inv;

			float2 d = { u_xy - u_pixf, v_xy - v_pixf }; 
			
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}