from mpi4py import MPI
import numpy as np
import os
import time
import sys


def mpi_process_vtk_files():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # Главный процесс собирает список файлов
        import glob
        vtk_files = glob.glob("*.vtk")
        print(f"Master found {len(vtk_files)} VTK files")

        # Распределяем файлы по процессам
        files_per_process = len(vtk_files) // size
        remainder = len(vtk_files) % size

        file_distribution = []
        start_idx = 0
        for i in range(size):
            end_idx = start_idx + files_per_process + (1 if i < remainder else 0)
            file_distribution.append(vtk_files[start_idx:end_idx])
            start_idx = end_idx
    else:
        file_distribution = None

    # Рассылаем распределение файлов
    local_files = comm.scatter(file_distribution, root=0)

    # Каждый процесс обрабатывает свои файлы
    local_results = []
    local_start_time = time.time()

    for filepath in local_files:
        try:
            # Импортируем здесь, чтобы избежать проблем с MPI
            from clusters_for_chains_from_vtk import clusterize_vtk, read_args
            cmd_args = read_args()

            file_start = time.time()
            clusters = clusterize_vtk(filepath, cmd_args.box_len)
            file_time = time.time() - file_start

            local_results.append({
                'file': filepath,
                'clusters': len(clusters),
                'time': file_time
            })
            print(f"Rank {rank}: Processed {filepath} in {file_time:.2f}s")

        except Exception as e:
            print(f"Rank {rank}: Error processing {filepath}: {e}")

    local_total_time = time.time() - local_start_time

    # Собираем результаты на главном процессе
    all_results = comm.gather(local_results, root=0)
    all_times = comm.gather(local_total_time, root=0)

    if rank == 0:
        print("\n" + "=" * 50)
        print("MPI PROCESSING RESULTS")
        print("=" * 50)

        total_files = 0
        total_clusters = 0
        max_time = 0

        for rank_results in all_results:
            for result in rank_results:
                total_files += 1
                total_clusters += result['clusters']
                print(f"File: {os.path.basename(result['file'])}")
                print(f"  Clusters: {result['clusters']}, Time: {result['time']:.2f}s")

        print(f"\nTotal files processed: {total_files}")
        print(f"Total clusters found: {total_clusters}")
        print(f"Maximum process time: {max(all_times):.2f}s")
        print(f"Average process time: {sum(all_times) / len(all_times):.2f}s")


if __name__ == "__main__":
    mpi_process_vtk_files()