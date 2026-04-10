def test_import():
    import modelb
    p = modelb.default_parameters()
    assert "n_cells" in p
