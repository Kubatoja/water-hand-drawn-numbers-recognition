"""
SZABLON 1: Konfiguracja optymalizacji Bayesian

Ten szablon pozwala na łatwą konfigurację:
- Wybór datasetów
- Parametry optymalizacji (iteracje, search space)
- Strategia (osobno dla każdego / uniwersalna)
"""

from Testers.BayesianOptimizer import (
    OptimizationOrchestrator,
    FULL_SEARCH_SPACE,
    QUICK_SEARCH_SPACE,
    # Datasety - dodaj/usuń według potrzeb
    MNIST_DATASET,
    EMNIST_BALANCED_DATASET,
    EMNIST_DIGITS_DATASET,
    ARABIC_DATASET,
    USPS_DATASET,
    # MNIST-C
    MNIST_C_BRIGHTNESS,
    MNIST_C_FOG,
    MNIST_C_ROTATE,
    # Kolekcje
    BASIC_DATASETS,
    DIGITS_ONLY_DATASETS,
    ALL_MNIST_C_DATASETS,
    # Helpers
    DatasetName,
    MnistCCorruption,
    get_dataset,
    get_datasets_by_names,
    create_mnist_c_config,
)


def main():
    """
    ========================================================================
    KONFIGURACJA OPTYMALIZACJI - EDYTUJ PONIŻSZE SEKCJE
    ========================================================================
    """
    
    # ========================================================================
    # 1. WYBÓR DATASETÓW
    # ========================================================================
    
    # OPCJA A: Ręczny wybór konkretnych datasetów
    datasets = [
        #MNIST_DATASET,
        #EMNIST_DIGITS_DATASET,
        ARABIC_DATASET,
        # USPS_DATASET,  # Odkomentuj aby dodać
    ]
    
    # OPCJA B: Użyj predefiniowanej kolekcji (zakomentuj OPCJA A, odkomentuj poniżej)
    # datasets = BASIC_DATASETS  # Wszystkie podstawowe (5)
    # datasets = DIGITS_ONLY_DATASETS  # Tylko 10-klasowe (4)
    
    # OPCJA C: Dodaj konkretne MNIST-C
    # datasets += [
    #     MNIST_C_BRIGHTNESS,
    #     MNIST_C_FOG,
    #     MNIST_C_ROTATE,
    # ]
    
    # OPCJA D: Dodaj wszystkie MNIST-C
    # datasets += ALL_MNIST_C_DATASETS  # Dodaje 16 wariantów
    
    # OPCJA E: Użyj enum do wyboru
    # datasets = get_datasets_by_names([
    #     DatasetName.MNIST,
    #     DatasetName.ARABIC,
    # ])
    
    # OPCJA F: Dynamiczne MNIST-C
    # mnist_c_variants = [
    #     MnistCCorruption.BRIGHTNESS,
    #     MnistCCorruption.FOG,
    #     MnistCCorruption.ROTATE,
    # ]
    # datasets += [create_mnist_c_config(c) for c in mnist_c_variants]
    
    
    # ========================================================================
    # 2. PARAMETRY OPTYMALIZACJI
    # ========================================================================
    
    # Search space - wybierz jeden:
    search_space = FULL_SEARCH_SPACE      # Pełna przestrzeń przeszukiwania
    # search_space = QUICK_SEARCH_SPACE   # Zredukowana (do szybkich testów)
    
    # Liczba iteracji dla każdego datasetu
    n_iterations = 20           # Zmień na 10-20 dla szybkich testów
    
    # Liczba losowych startów (exploration)
    n_random_starts = 5        # Zazwyczaj 10-20% z n_iterations
    
    # Tryb verbose (szczegółowe logi)
    verbose = True
    
    
    # ========================================================================
    # 3. OPCJE ZAAWANSOWANE (opcjonalne)
    # ========================================================================
    
    # Czy zapisywać wyniki po każdym teście?
    save_after_each = True  # True = bezpieczniejsze (masz partial results)
    
    # Czy pomijać pierwszą generację wektorów? (jeśli już istnieją)
    skip_first_vectors = False
    
    
    # ========================================================================
    # 4. WALIDACJA I PODSUMOWANIE
    # ========================================================================
    
    if not datasets:
        print("BŁĄD: Nie wybrano żadnych datasetów!")
        print("Odkomentuj jedną z OPCJI w sekcji 'WYBÓR DATASETÓW'")
        return
    
    print("=" * 80)
    print("KONFIGURACJA OPTYMALIZACJI BAYESIAN")
    print("=" * 80)
    
    print(f"\n📊 Datasety ({len(datasets)}):")
    for i, ds in enumerate(datasets, 1):
        print(f"  {i:2d}. {ds.display_name:40s} ({ds.class_count} classes, {ds.image_size}x{ds.image_size})")
    
    print(f"\n⚙️  Parametry optymalizacji:")
    print(f"  Search space:     {'FULL' if search_space == FULL_SEARCH_SPACE else 'QUICK'}")
    print(f"  Iterations:       {n_iterations} per dataset")
    print(f"  Random starts:    {n_random_starts}")
    print(f"  Verbose:          {verbose}")
    print(f"  Save after each:  {save_after_each}")
    
    print(f"\n📈 Szacunki:")
    total_runs = len(datasets) * n_iterations
    print(f"  Total runs:       {total_runs}")
    print(f"  Estimated time:   {total_runs * 1:.0f} - {total_runs * 3:.0f} minut")
    print(f"                    ({total_runs * 1 / 60:.1f} - {total_runs * 3 / 60:.1f} godzin)")
    
    print("=" * 80)
    
    
    # ========================================================================
    # 5. POTWIERDZENIE I URUCHOMIENIE
    # ========================================================================

    
    print("\n🚀 Uruchamianie optymalizacji...")
    print("=" * 80)
    
    # Konfiguracja test runnera
    from Testers.Shared.configs import TestRunnerConfig
    test_runner_config = TestRunnerConfig(
        skip_first_vector_generation=skip_first_vectors,
        save_results_after_each_test=save_after_each
    )
    
    # Utwórz orchestrator
    orchestrator = OptimizationOrchestrator(
        datasets=datasets,
        search_space_config=search_space,
        n_iterations=n_iterations,
        n_random_starts=n_random_starts,
        test_runner_config=test_runner_config,
        verbose=verbose
    )
    
    # Uruchom optymalizację
    results = orchestrator.run_optimization()
    
    
    # ========================================================================
    # 6. PODSUMOWANIE WYNIKÓW
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("✅ OPTYMALIZACJA ZAKOŃCZONA")
    print("=" * 80)
    print(f"\nZoptymalizowano {len(results)} datasetów:\n")
    
    for dataset_name, result in results.items():
        print(f"📊 {dataset_name}")
        print(f"   Best accuracy: {result.best_score:.4f}")
        print(f"   Best params:")
        for param, value in result.best_params.items():
            print(f"      {param:30s} = {value}")
        print()
    
    print("=" * 80)
    print("📁 Wyniki zapisane w: results/")
    print("=" * 80)


if __name__ == "__main__":
    """
    ========================================================================
    QUICK START GUIDE:
    ========================================================================
    
    1. Edytuj sekcję "1. WYBÓR DATASETÓW" - wybierz datasety do testowania
    2. Edytuj sekcję "2. PARAMETRY OPTYMALIZACJI" - ustaw iteracje i search space
    3. Uruchom: python template_optimization.py
    4. Sprawdź wyniki w folderze: results/
    
    ========================================================================
    PRZYKŁADOWE KONFIGURACJE:
    ========================================================================
    
    SZYBKI TEST (5-10 minut):
    - datasets = [MNIST_DATASET]
    - search_space = QUICK_SEARCH_SPACE
    - n_iterations = 10
    
    ŚREDNI TEST (30-60 minut):
    - datasets = DIGITS_ONLY_DATASETS
    - search_space = FULL_SEARCH_SPACE
    - n_iterations = 30
    
    PEŁNA OPTYMALIZACJA (kilka godzin):
    - datasets = BASIC_DATASETS + wybrane MNIST-C
    - search_space = FULL_SEARCH_SPACE
    - n_iterations = 50
    
    ========================================================================
    """
    main()
