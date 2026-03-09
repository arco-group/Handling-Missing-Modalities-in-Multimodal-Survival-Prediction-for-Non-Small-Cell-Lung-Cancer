from CMC_utils.miscellaneous import recursive_cfg_substitute
import pandas as pd
import numpy as np
from lifelines.utils import concordance_index

# Importazioni per metriche avanzate (scikit-survival)
try:
    from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc, integrated_brier_score as sksurv_ibs
    from sksurv.util import Surv

    SKSURV_AVAILABLE = True
except ImportError:
    SKSURV_AVAILABLE = False
    print("Attenzione: scikit-survival non installato. Le metriche 'uno_c_index', 'td_auc' e 'ibs' non funzioneranno.")

__all__ = ["Ct_index", "set_ct_index_params", "c_index", "uno_c_index", "td_auc", "integrated_brier_score"]


def set_ct_index_params(metrics: dict, preprocessing_params: dict):
    """
    Set the parameters of the Concordance index
    """
    metrics = recursive_cfg_substitute(metrics, dict(num_events=preprocessing_params["num_events"]))
    # Se necessario, si possono passare anche i tempi massimi o gli intervalli qui
    return metrics


# --- Helper Functions ---

def _to_structured_array(y_true: np.ndarray) -> np.ndarray:
    """
    Converte l'array numpy [Event, Time] nel formato strutturato per scikit-survival.
    """
    events = y_true[:, 0].astype(bool)
    times = y_true[:, 1]
    return Surv.from_arrays(event=events, time=times)


def _prepare_risk_score(y_score: np.ndarray) -> np.ndarray:
    """Aggrega lo score temporale in un unico risk score scalare (per C-index)"""
    return (np.sum(y_score, axis=1) / y_score.shape[1])


def _prepare_survival_curves(y_score: np.ndarray, num_events: int = 1) -> np.ndarray:
    """
    Converte l'input y_score (spesso CIF appiattita) in curve di sopravvivenza S(t).
    Assumiamo che y_score sia la Cumulative Incidence Function (CIF).
    S(t) = 1 - CIF(t)
    """
    batch_dim = y_score.shape[0]
    # Reshape basato sulla logica di Ct_index
    # Se y_score è piatto (batch, time * events), lo riportiamo a (batch, time) assumendo evento principale
    if num_events > 1:
        # Se ci sono rischi competitivi, prendiamo l'evento di interesse (es. indice 0)
        # Nota: la struttura esatta dipende da come il tuo modello emette i dati.
        # Qui assumiamo la stessa logica di reshape di Ct_index:
        CIF_all = y_score.reshape((batch_dim, num_events, -1))
        CIF = CIF_all[:, 0, :]  # Prendiamo il primo evento come "evento di interesse"
    else:
        CIF = y_score  # Già (batch, time)

    # Brier Score vuole probabilità di SOPRAVVIVENZA, non di evento.
    survival_curves = 1.0 - CIF
    return survival_curves


# --- Core Metrics ---

def Ct_index(y_true: np.ndarray, y_score: np.ndarray, num_events: int, average: bool = False, **kwargs) -> float:
    # ... (Il tuo codice Ct_index originale rimane invariato) ...
    def get_ct_index_mask(labels: np.ndarray) -> np.ndarray:
        batch_dim = labels.shape[0]
        tmp1 = np.repeat(np.expand_dims(labels[:, 1], axis=1), batch_dim, axis=1)
        tmp1 = tmp1 < tmp1.transpose(1, 0)
        tmp2 = np.repeat(np.expand_dims(labels[:, 0], axis=1), batch_dim, axis=1)
        tmp2 = tmp2 != 0
        mask = (tmp1 * tmp2).astype(int)
        return mask

    batch_dim = y_true.shape[0]
    CIF = y_score.reshape((batch_dim, num_events, -1))
    sample_idx = np.arange(batch_dim)
    k_event_idx = np.clip(y_true[:, 0] - 1, a_min=0, a_max=None)
    k_time_idx = y_true[:, 1]
    tmp1 = CIF[sample_idx, k_event_idx, k_time_idx]
    tmp1 = np.repeat(np.expand_dims(tmp1, axis=1), batch_dim, axis=1)
    CIF_ref = np.swapaxes(CIF, 2, 0)
    tmp2 = CIF_ref[k_time_idx, k_event_idx]
    tmp = (tmp1 > tmp2).astype(int)
    mask = get_ct_index_mask(y_true)
    tmp_num = np.repeat(np.sum(mask * tmp, axis=1, keepdims=True), num_events, axis=1)
    tmp_den = np.repeat(np.sum(mask, axis=1, keepdims=True), num_events, axis=1)
    k_masks = np.vstack([y_true[:, 0] == k for k in range(1, num_events + 1)]).transpose(1, 0)
    tmp_num = k_masks * tmp_num
    tmp_den = k_masks * tmp_den
    tmp_num = np.sum(tmp_num, axis=0)
    tmp_den = np.sum(tmp_den, axis=0)
    ct_index = tmp_num / tmp_den

    if average:
        return np.sum(ct_index)
    return ct_index
# --- 1. Helper Functions (Assicurati che siano definite prima di integrated_brier_score) ---

def _to_structured_array(y_true: np.ndarray) -> np.ndarray:
    """Converte l'array numpy [Event, Time] nel formato strutturato per scikit-survival."""
    events = y_true[:, 0].astype(bool)
    times = y_true[:, 1]
    return Surv.from_arrays(event=events, time=times)

def _prepare_survival_curves(y_score: np.ndarray, num_events: int = 1) -> np.ndarray:
    """
    Converte CIF (Cumulative Incidence) in curve di Sopravvivenza S(t).
    S(t) = 1 - CIF(t)
    """
    batch_dim = y_score.shape[0]
    if num_events > 1:
        # Se ci sono rischi competitivi, reshape e prendi l'evento di interesse
        CIF_all = y_score.reshape((batch_dim, num_events, -1))
        CIF = CIF_all[:, 0, :] 
    else:
        CIF = y_score 
    
    survival_curves = 1.0 - CIF
    return survival_curves



# --- 2. Funzione Integrated Brier Score (Corretta e Robusta) ---

def integrated_brier_score(y_true, y_score, **kwargs):
    """
    Integrated Brier Score (IBS) - Versione Robusta per scikit-survival.
    """
    # Controllo disponibilità libreria
    if not SKSURV_AVAILABLE:
        return -1.0

    # Conversione dati
    y_true_struct = _to_structured_array(y_true)
    
    # Recupero dati training (o fallback su y_true se non forniti)
    y_train = kwargs.get('y_train_struct', y_true_struct)
    
    # Preparazione curve
    num_events = kwargs.get('num_events', 1)
    surv_probs = _prepare_survival_curves(y_score, num_events=num_events)
    
    # --- Gestione Griglia Temporale (Cruciale per evitare errori) ---
    n_times = surv_probs.shape[1]
    
    if 'times' in kwargs:
        times = kwargs['times']
    else:
        # Creiamo una griglia temporale fittizia basata sul range dei dati reali
        t_max_data = y_true_struct["time"].max()
        t_min_data = y_true_struct["time"].min()
        times = np.linspace(t_min_data, t_max_data, n_times)

    # --- Filtraggio Validità (Range [min, max]) ---
    t_min = y_true_struct["time"].min()
    t_max = y_true_struct["time"].max()
    
    # Maschera: teniamo solo i tempi che cadono nel range osservato
    valid_mask = (times >= t_min) & (times <= t_max)
    
    eval_times = times[valid_mask]
    eval_surv = surv_probs[:, valid_mask]

    # --- Controllo Anti-Crash (Minimo 2 punti) ---
    if len(eval_times) < 2:
        return np.nan

    try:
        # Calcolo effettivo
        score = sksurv_ibs(y_train, y_true_struct, eval_surv, eval_times)
        return score
    except ValueError:
        return np.nan
    except Exception as e:
        print(f"Errore IBS: {e}")
        return np.nan


def integrated_brier_score(y_true, y_score, **kwargs):
    """
    Integrated Brier Score (IBS).

    Misura l'accuratezza delle probabilità predette (Calibrazione + Discriminazione).
    Valori più bassi sono migliori (0 = perfetto, 0.25 = random per binario bilanciato).

    Richiede:
    - y_train_struct in kwargs (per stimare la censura IPCW).
    - time_grid: i punti temporali corrispondenti alle colonne di y_score.
    """
    if not SKSURV_AVAILABLE:
        return -1.0

    y_true_struct = _to_structured_array(y_true)

    # Recuperiamo i dati di training per la stima della censura (fondamentale per IBS)
    # Se non forniti, usiamo y_true come fallback (meno rigoroso ma funzionante)
    y_train = kwargs.get('y_train_struct', y_true_struct)

    # Recuperiamo il numero di eventi e prepariamo le curve S(t)
    num_events = kwargs.get('num_events', 1)
    surv_probs = _prepare_survival_curves(y_score, num_events=num_events)

    # Definizione della griglia temporale
    # y_score ha shape (N, T). Dobbiamo sapere a quali tempi "t" corrispondono le colonne.
    # Se non passato, assumiamo che i tempi siano interi da 1 a T (come nel tuo esempio dummy)
    n_times = surv_probs.shape[1]
    times = kwargs.get('times', np.arange(1, n_times + 1))

    # Tagliamo i tempi per stare dentro il range del test set (richiesto da sksurv)
    # sksurv richiede che i tempi di valutazione siano entro min e max di y_test
    t_min = y_true_struct["time"].min()
    t_max = y_true_struct["time"].max()

    # Maschera per selezionare solo i tempi validi e le colonne corrispondenti di surv_probs
    valid_mask = (times >= t_min) & (times < t_max)

    if not np.any(valid_mask):
        # Fallback se nessun tempo corrisponde (caso raro/edge case)
        return np.nan

    eval_times = times[valid_mask]
    eval_surv = surv_probs[:, valid_mask]

    # Calcolo IBS
    score = sksurv_ibs(y_train, y_true_struct, eval_surv, eval_times)
    return score


def uno_c_index(y_true, y_score, **kwargs):
    """Uno's C-index (Robust to Censoring)"""
    if not SKSURV_AVAILABLE:
        return -1.0
    y_true_struct = _to_structured_array(y_true)
    risk_score = _prepare_risk_score(y_score)
    y_train = kwargs.get('y_train_struct', y_true_struct)
    uno_c, _, _, _, _ = concordance_index_ipcw(y_train, y_true_struct, risk_score)
    return uno_c


def td_auc(y_true, y_score, **kwargs):
    """Time-Dependent AUC (Mean over time)"""
    if not SKSURV_AVAILABLE:
        return -1.0
    y_true_struct = _to_structured_array(y_true)
    risk_score = _prepare_risk_score(y_score)
    y_train = kwargs.get('y_train_struct', y_true_struct)

    # Valutiamo ai percentili 25, 50, 75 dei tempi osservati
    time_range = np.percentile(y_true_struct["time"], [25, 50, 75])

    try:
        auc, mean_auc = cumulative_dynamic_auc(y_train, y_true_struct, risk_score, time_range)
        return mean_auc
    except ValueError:
        return np.nan  # Può fallire se i tempi sono fuori range


def c_index(y_true, y_score, **kwargs):
    """Harrell's C-index (Original)"""
    y_score = (np.sum(y_score, axis=1, keepdims=True) / y_score.shape[1])
    y_pred = pd.DataFrame({"prediction": y_score.tolist()})
    y_true = pd.DataFrame({"efs": y_true[:, 0].tolist(), "efs_time": y_true[:, 1].tolist()})
    y_true.insert(0, "ID", range(len(y_true)))
    y_pred.insert(0, "ID", range(len(y_pred)))
    y_pred.loc[:, "prediction"] = y_pred["prediction"].explode()
    a = score(y_true, y_pred.astype(float), "ID")
    return a


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    del solution[row_id_column_name]
    del submission[row_id_column_name]
    merged_df = pd.concat([solution, submission], axis=1)
    merged_df.reset_index(inplace=True)
    return concordance_index(merged_df['efs_time'], -merged_df['prediction'], merged_df['efs'])

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Helper originale per Harrell's C-index
    """
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    event_label = 'efs'
    interval_label = 'efs_time'
    prediction_label = 'prediction'

    merged_df = pd.concat([solution, submission], axis=1)
    merged_df.reset_index(inplace=True)

    c_index_val = concordance_index(
            merged_df[interval_label],
            -merged_df[prediction_label],  # Nota il segno meno: lifelines assume rischio alto = tempo breve
            merged_df[event_label])

    return c_index_val


if __name__ == "__main__":
    # Esempio Dummy
    y_true_dummy = np.array([[1, 10], [0, 20], [1, 5]])  # Evento, Tempo
    y_score_dummy = np.array([[0.9, 0.2], [0.4, 0.5], [0.1, 0.2]])  # Score grezzi
    brier_score = integrated_brier_score(y_true_dummy, y_score_dummy, times=np.array([1, 5, 10, 15, 20]), num_events=1, y_train_struct=_to_structured_array(y_true_dummy))

    print(f"Harrell C-index: {c_index(y_true_dummy, y_score_dummy)}")
    print(f"Integrated Brier Score: {brier_score}")

    try:
        print(f"Uno C-index: {uno_c_index(y_true_dummy, y_score_dummy)}")
        print(f"Time-Dependent AUC (Mean): {td_auc(y_true_dummy, y_score_dummy)}")
    except Exception as e:
        print(f"Errore nel calcolo metriche avanzate: {e}")