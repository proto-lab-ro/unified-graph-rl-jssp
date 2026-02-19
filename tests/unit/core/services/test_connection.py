from pathlib import Path

import pytest
import yaml


try:
    import optuna
    import pyodbc

    PYODBC_AVAILABLE = True
except ImportError:
    PYODBC_AVAILABLE = False


# Fixture for credentials
@pytest.fixture(scope="session")
def credentials():
    credentials_path = Path("./src/hpo/credentials.yaml")
    if not credentials_path.exists():
        pytest.skip("Credentials file not found")
    with open(credentials_path) as f:
        return yaml.safe_load(f)


# Skip all tests in this module if pyodbc is not available
pytestmark = pytest.mark.skipif(
    not PYODBC_AVAILABLE, reason="pyodbc or dependencies (libodbc) not available"
)


# Fixture for database connection parameters
@pytest.fixture(scope="session")
def db_params(credentials):
    return {
        "driver": credentials["driver"],
        "server": credentials["host"],
        "database": credentials["database"],
        "username": credentials["user"],
        "password": credentials["password"],
    }


# Test credentials file
def test_credentials_file_exists(credentials):
    """Test if credentials file has required fields."""
    required_fields = ["host", "database", "user", "password", "driver"]
    for field in required_fields:
        assert field in credentials


# Test database connection
def test_db_connection(db_params):
    """Test direct database connection using pyodbc."""
    conn = pyodbc.connect(
        f"DRIVER={db_params['driver']};"
        f"SERVER={db_params['server']};"
        f"DATABASE={db_params['database']};"
        f"UID={db_params['username']};"
        f"PWD={db_params['password']}"
    )
    cursor = conn.cursor()

    cursor.execute("SELECT GETDATE();")
    result = cursor.fetchone()
    assert result is not None

    cursor.close()
    conn.close()


# Test Optuna connection
def test_optuna_connection(db_params):
    """Test Optuna connection and basic functionality."""
    storage_url = (
        f"mssql+pyodbc://{db_params['username']}:{db_params['password']}"
        f"@{db_params['server']}/{db_params['database']}"
        f"?driver={db_params['driver'].replace(' ', '+')}"
    )

    study = optuna.create_study(
        study_name="pytest_test_study", storage=storage_url, load_if_exists=True
    )

    def objective(trial):
        x = trial.suggest_float("x", -10, 10)
        return (x - 2) ** 2

    study.optimize(objective, n_trials=1)

    assert study.best_trial is not None
    assert study.best_value is not None
    assert study.best_params is not None


# Test database connection failure
def test_db_connection_failure(db_params, mocker):
    """Test database connection failure handling."""
    mocker.patch("pyodbc.connect", side_effect=pyodbc.Error("Connection failed"))

    with pytest.raises(pyodbc.Error):
        pyodbc.connect(
            f"DRIVER={db_params['driver']};"
            f"SERVER={db_params['server']};"
            f"DATABASE={db_params['database']};"
            f"UID={db_params['username']};"
            f"PWD={db_params['password']}"
        )
