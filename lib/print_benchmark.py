import json
import sys
import os


def print_benchmark_table(json_path: str = "benchmark_results.json") -> None:
    """
    Читает JSON-файл с результатами бенчмарка и выводит
    итоговую таблицу в консоль в табличном виде.
    """
    if not os.path.exists(json_path):
        print(f"Файл '{json_path}' не найден.")
        print("   Сначала запустите основной скрипт бенчмарка,")
        print("   чтобы сгенерировать benchmark_results.json.")
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    sep = "=" * 90
    dash = "-" * 90

    print(sep)
    print("ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print(sep)

    header = (
        f"{'Конфиг':<10} {'Vocab':>6} {'d_mod':>6} {'Layers':>7} "
        f"{'Batch':>6} {'SeqIn':>6} {'NewTok':>7} "
        f"{'Torch(s)':>10} {'NumPy(s)':>10} {'Speedup':>8}x {'Match':>6}"
    )
    print(header)
    print(dash)

    for r in results:
        p = r["params"]
        match_str = "OK" if r["results_match"] else "FAIL"
        row = (
            f"{r['config']:<10} "
            f"{p['vocab_size']:>6} "
            f"{p['d_model']:>6} "
            f"{p['n_layers']:>7} "
            f"{p['batch_size']:>6} "
            f"{p['seq_len_input']:>6} "
            f"{p['max_new_tokens']:>7} "
            f"{r['time_torch_sec']:>10.6f} "
            f"{r['time_numpy_sec']:>10.6f} "
            f"{r['speedup']:>8.2f}x "
            f"{match_str:>6}"
        )
        print(row)

    print(sep)


if __name__ == "__main__":
    # Опционально можно передать путь к JSON через аргумент командной строки:
    #   python print_benchmark.py path/to/benchmark_results.json
    path = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results.json"
    print_benchmark_table(path)