import argparse
import collections 
import copy
import os
from pathlib import Path
import time

import igraph as ig
import numpy as np
from scipy.spatial import distance
import vtk
from vtk.util.numpy_support import vtk_to_numpy
def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename',
        type=str, help='path to vtk file from simulations',
        default='1.L.5.fl.10.fn.512.dist.0.05.k.10.dat.step.33800.vtk')
    parser.add_argument('-bl', '--box_len', type=float, help='Box Length', default=46.784)
    parser.add_argument('-fl', '--fl', type=int, help='Filament len', default=10)
    parser.add_argument('-dth', '--distance_threshold', type=float, help='Distance threshold', default=1.2)
    parser.add_argument('-s', '--single', action='store_true', help='Process single file')
    parser.add_argument('-p', '--parallel', action='store_true', help='Process all files in parallel')
    parser.add_argument('-mw', '--max_workers', type=int, help='Max workers for threading', default=4)
    cmd_args = parser.parse_args()
    return cmd_args

def print_elapsed_time(start):
    end = time.time()
    m, s = divmod(end - start, 60)
    h, m = divmod(m, 60)
    print('Elapsed time  = {0:2.0f}h:{1:2.0f}m:{2:2.0f}s'.format(h, m, s))

def periodic_pdist(N, coords, boxl):
   # https://github.com/Allen-Tildesley/examples/blob/master/python_examples/md_lj_module.py#L121
    dim = 3

    pair_shift = np.empty((dim, N, N))
    pos_in_box = coords / boxl  # convert the unit to box

    for d in range(dim):
        pos_1d = pos_in_box[:, d][:, np.newaxis]  # shape (N, 1),  make it as column vector by inserting an axis along second dimension
        pair_shift_1d = pos_1d - pos_1d.T  # shape (N, N)
        pair_shift_1d = pair_shift_1d - np.rint(pair_shift_1d)
        pair_shift[d] = pair_shift_1d
    
    dist_nd = np.linalg.norm(pair_shift, axis=0) * boxl  # conver the unit back
    return dist_nd

def get_info_from_vtk(vtk_filename):
    """
    load a vtk file as input and return 4 arrays
    coordinates of points, bonds, orientation, number of initial chain
    """
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(vtk_filename)
    reader.Update()

    coords_vtk = reader.GetOutput().GetPoints().GetData()
    coords = vtk_to_numpy(coords_vtk)
    # print('coords', coords)

    cell_vtk = reader.GetOutput().GetCells().GetData()
    bonds = vtk_to_numpy(cell_vtk)
    bonds.resize(int(len(bonds)/3), 3) # all bonds are three number: 2 part_id1 part_id2
    bonds_per_part = collections.defaultdict(list)
    for t, p1, p2 in bonds:
        bonds_per_part[p1].append((p1, p2))
        bonds_per_part[p2].append((p1, p2))

    mag_moments_vtk = reader.GetOutput().GetPointData().GetArray('mag_moments')
    mag_moments = vtk_to_numpy(mag_moments_vtk)

    cluster_num_vtk = reader.GetOutput().GetPointData().GetArray('cluster_num')
    cluster_num = vtk_to_numpy(cluster_num_vtk)
    return coords, mag_moments, bonds_per_part, cluster_num

def clusterize_vtk(vtk_filename, boxl, dth=1.2):
    
    coords, mag_moments, bonds_per_part, cluster_num = get_info_from_vtk(vtk_filename)
    
    # print('cluster_num', cluster_num)

    N = len(coords)
    print(f'Number of particles is {N}')
    
    pp_dist = periodic_pdist(N, coords, boxl)
    edges = []
    # Создаем список ребер на основе критерия кластеризации
    for i in range(N):
        for j in range(i + 1, N):
            # Критерий: расстояние <= порога ИЛИ одинаковый cluster_num
            if pp_dist[i, j] <= dth or cluster_num[i] == cluster_num[j]:
                edges.append((i, j))

    g = ig.Graph()
    g.add_vertices([i for i in range(N)])
    g.add_edges(edges)

    graph_clusters = g.clusters()
    # Создаем директорию для сохранения кластеров
    base_name = os.path.splitext(os.path.basename(vtk_filename))[0]
    output_dir = f"clusters_{base_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Сохраняем каждый кластер в отдельный файл
    for cluster_idx, cluster_vertices in enumerate(graph_clusters):
        if len(cluster_vertices) > 0:
            # Получаем координаты частиц в кластере
            cluster_coords = coords[cluster_vertices]

            # Создаем имя файла для кластера
            cluster_filename = os.path.join(output_dir, f"cluster_{cluster_idx}_size_{len(cluster_vertices)}.vtk")

            # Создаем VTK файл для кластера
            points = vtk.vtkPoints()
            for coord in cluster_coords:
                points.InsertNextPoint(coord[0], coord[1], coord[2])

            # Создаем полидату
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points)

            # Добавляем вершины как VTK вершины
            vertices = vtk.vtkCellArray()
            for i in range(len(cluster_coords)):
                vertex = vtk.vtkVertex()
                vertex.GetPointIds().SetId(0, i)
                vertices.InsertNextCell(vertex)

            polydata.SetVerts(vertices)

            # Записываем в файл
            writer = vtk.vtkPolyDataWriter()
            writer.SetFileName(cluster_filename)
            writer.SetInputData(polydata)
            writer.Write()

            print(f"Saved cluster {cluster_idx} with {len(cluster_vertices)} particles to {cluster_filename}")

    print(f"Total clusters found: {len(graph_clusters)}")
    return graph_clusters

def fold_coordinates(particles_pos, box_size):
    folded_pos = copy.deepcopy(particles_pos)
    half_box_size = 0.5 * box_size
    N = particles_pos.shape[0]
    for dim in range(3):
        one_dim_slice = (folded_pos[:, dim]).reshape((N, 1))
        distance_list = distance.pdist(one_dim_slice)
        distance_matr = distance.squareform(distance_list)
        # print(distance_list)

        positive_mask = np.nonzero(distance_matr >= half_box_size)

        for k in range(len(positive_mask[0])):
            i = positive_mask[0][k]
            j = positive_mask[1][k]
            if particles_pos[i, dim] < particles_pos[j, dim]:
                folded_pos[i, dim] = box_size + particles_pos[i, dim]
    return folded_pos


import threading
import concurrent.futures
from multiprocessing import Pool, cpu_count
import glob


def process_vtk_file_single(filepath, boxl, dth=1.2):
    """Обработка одного VTK файла"""
    start_time = time.time()
    print(f"Processing {filepath}")
    clusters = clusterize_vtk(filepath, boxl, dth)
    elapsed = time.time() - start_time
    print(f"Completed {filepath} in {elapsed:.2f} seconds")
    return len(clusters), elapsed


def process_vtk_files_threading(vtk_files, boxl, dth=1.2, max_workers=4):
    """Параллельная обработка с использованием threading"""
    start_time = time.time()
    results = []

    def worker(filepath):
        return process_vtk_file_single(filepath, boxl, dth)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, filepath): filepath for filepath in vtk_files}

        for future in concurrent.futures.as_completed(futures):
            filepath = futures[future]
            try:
                num_clusters, file_time = future.result()
                results.append((filepath, num_clusters, file_time))
            except Exception as e:
                print(f"Error processing {filepath}: {e}")

    total_time = time.time() - start_time
    return results, total_time


def process_vtk_files_multiprocessing(vtk_files, boxl, dth=1.2):
    """Параллельная обработка с использованием multiprocessing"""
    start_time = time.time()

    # Создаем список аргументов для каждого процесса
    args_list = [(filepath, boxl, dth) for filepath in vtk_files]

    # Используем Pool для параллельной обработки
    with Pool(processes=min(cpu_count(), len(vtk_files))) as pool:
        results = pool.starmap(process_vtk_file_single, args_list)

    total_time = time.time() - start_time

    # Формируем результаты с именами файлов
    detailed_results = []
    for i, (num_clusters, file_time) in enumerate(results):
        detailed_results.append((vtk_files[i], num_clusters, file_time))

    return detailed_results, total_time


def main_parallel_processing():
    """Основная функция для параллельной обработки"""
    cmd_args = read_args()

    # Находим все VTK файлы в текущей директории
    vtk_files = glob.glob("*.vtk")
    if not vtk_files:
        print("No VTK files found in current directory")
        return

    print(f"Found {len(vtk_files)} VTK files")

    # Обработка с threading
    print("\n" + "=" * 50)
    print("Processing with THREADING")
    print("=" * 50)
    threading_results, threading_time = process_vtk_files_threading(
        vtk_files, cmd_args.box_len, max_workers=4
    )

    # Обработка с multiprocessing
    print("\n" + "=" * 50)
    print("Processing with MULTIPROCESSING")
    print("=" * 50)
    multiprocessing_results, multiprocessing_time = process_vtk_files_multiprocessing(
        vtk_files, cmd_args.box_len
    )

    # Вывод результатов
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)

    print(f"\nThreading total time: {threading_time:.2f} seconds")
    print(f"Multiprocessing total time: {multiprocessing_time:.2f} seconds")

    # Сравнение производительности
    if threading_time > 0 and multiprocessing_time > 0:
        speedup = threading_time / multiprocessing_time
        print(f"\nSpeedup (Multiprocessing vs Threading): {speedup:.2f}x")

    # Детальная статистика
    print("\nDetailed statistics per file:")
    for i, (filepath, _, _) in enumerate(vtk_files[:5]):  # Показываем первые 5 файлов
        if i < len(threading_results):
            print(f"\nFile: {os.path.basename(filepath)}")
            print(f"  Threading time: {threading_results[i][2]:.2f}s, Clusters: {threading_results[i][1]}")
            print(
                f"  Multiprocessing time: {multiprocessing_results[i][2]:.2f}s, Clusters: {multiprocessing_results[i][1]}")

if __name__ == '__main__':
    cmd_args = read_args()
    start = time.time()
    clusterize_vtk(cmd_args.filename, cmd_args.box_len)
    print_elapsed_time(start)
    # Одиночная обработка
    if hasattr(cmd_args, 'single') and cmd_args.single:
        start = time.time()
        clusterize_vtk(cmd_args.filename, cmd_args.box_len)
        print_elapsed_time(start)

    # Параллельная обработка
    elif hasattr(cmd_args, 'parallel') and cmd_args.parallel:
        main_parallel_processing()


