"""
BioMission Suite - Plateforme de bio-informatique
Version 2.0 - Single File Deployable
Auteur : Slimane Rahim
Déployable sur Streamlit Community Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO, BytesIO
import gzip
import os
import tempfile
from datetime import datetime

# ============================================================
# CONFIGURATION DE LA PAGE
# ============================================================
st.set_page_config(
    page_title="BioMission Suite",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# IMPORTS OPTIONNELS (dégradation gracieuse)
# ============================================================
BIO_AVAILABLE = False
POD5_AVAILABLE = False
SKLEARN_AVAILABLE = False
XGB_AVAILABLE = False
LGBM_AVAILABLE = False
SHAP_AVAILABLE = False
JOBLIB_AVAILABLE = False

try:
    from Bio import SeqIO
    BIO_AVAILABLE = True
except ImportError:
    pass

try:
    import pod5
    POD5_AVAILABLE = True
except ImportError:
    pass

try:
    from sklearn.ensemble import (
        RandomForestClassifier, RandomForestRegressor,
        GradientBoostingClassifier, GradientBoostingRegressor
    )
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import (
        train_test_split, StratifiedKFold, KFold,
        cross_validate, GridSearchCV
    )
    from sklearn.metrics import (
        classification_report, roc_curve, auc,
        confusion_matrix, mean_squared_error, r2_score,
        mean_absolute_error
    )
    from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.inspection import permutation_importance
    SKLEARN_AVAILABLE = True
except ImportError:
    pass

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    pass

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    pass

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    pass

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    pass

# ============================================================
# STYLES CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
        padding: 1rem 0;
    }
    .module-title {
        color: #1f4e79;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
    }
    .stAlert {
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# PARSERS
# ============================================================
def read_plink_ped_map(ped_content, map_content):
    """Lit PED/MAP et retourne samples, variants, genotypes."""
    map_cols = ["chrom", "snp_id", "cm", "pos"]
    variants = pd.read_csv(
        StringIO(map_content), sep=r"\s+", header=None, names=map_cols
    )

    n_variants = len(variants)
    meta_cols = ["fid", "iid", "pid", "mid", "sex", "phenotype"]
    geno_cols = [f"g{i}" for i in range(2 * n_variants)]

    df = pd.read_csv(
        StringIO(ped_content), sep=r"\s+", header=None,
        names=meta_cols + geno_cols, dtype=str
    )

    samples = df[meta_cols].copy()
    genotypes = df[geno_cols].copy()
    return samples, variants, genotypes


def parse_fastq_simple(content, max_reads=None):
    """Parse FASTQ sans Biopython (fallback)."""
    reads = []
    lines = content.split("\n")
    i = 0
    count = 0
    while i < len(lines) - 3:
        if lines[i].startswith("@"):
            read_id = lines[i][1:].strip()
            seq = lines[i + 1].strip()
            plus = lines[i + 2]
            qual = lines[i + 3].strip() if i + 3 < len(lines) else ""
            if plus.startswith("+"):
                reads.append({
                    "read_id": read_id,
                    "sequence": seq,
                    "length": len(seq),
                    "quality": qual
                })
                count += 1
                if max_reads and count >= max_reads:
                    break
                i += 4
            else:
                i += 1
        else:
            i += 1
    return reads


def parse_fastq_biopython(content, max_reads=None):
    """Parse FASTQ avec Biopython."""
    records = []
    for i, rec in enumerate(SeqIO.parse(StringIO(content), "fastq")):
        if max_reads and i >= max_reads:
            break
        records.append({
            "read_id": rec.id,
            "sequence": str(rec.seq),
            "length": len(rec.seq),
            "quality": rec.letter_annotations.get("phred_quality", [])
        })
    return records


def parse_vcf_simple(content, max_variants=None):
    """Parse VCF sans dépendances externes."""
    variants = []
    count = 0
    for line in content.split("\n"):
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 8:
            try:
                qual = float(parts[5]) if parts[5] != "." else 0.0
            except ValueError:
                qual = 0.0
            try:
                pos = int(parts[1])
            except ValueError:
                pos = 0
            variants.append({
                "CHROM": parts[0],
                "POS": pos,
                "ID": parts[2],
                "REF": parts[3],
                "ALT": parts[4],
                "QUAL": qual,
                "FILTER": parts[6],
                "INFO": parts[7][:100]
            })
            count += 1
            if max_variants and count >= max_variants:
                break
    return variants


def read_fastq_file(uploaded_file, max_reads=10000):
    """Lit un fichier FASTQ (gz ou non)."""
    raw = uploaded_file.read()
    if uploaded_file.name.endswith(".gz"):
        content = gzip.decompress(raw).decode("utf-8", errors="ignore")
    else:
        content = raw.decode("utf-8", errors="ignore")

    if BIO_AVAILABLE:
        return parse_fastq_biopython(content, max_reads)
    return parse_fastq_simple(content, max_reads)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.markdown("# 🧬 BioMission Suite")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Accueil",
        "📤 Upload",
        "🔍 QC FASTQ",
        "🧬 PLINK (PED/MAP)",
        "🔬 Nanopore (POD5/FAST5)",
        "📊 VCF Variants",
        "🤖 Machine Learning",
        "📄 Rapports"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Statut bibliothèques")
st.sidebar.markdown(f"- Biopython : {'✅' if BIO_AVAILABLE else '❌'}")
st.sidebar.markdown(f"- pod5 : {'✅' if POD5_AVAILABLE else '❌'}")
st.sidebar.markdown(f"- scikit-learn : {'✅' if SKLEARN_AVAILABLE else '❌'}")
st.sidebar.markdown(f"- XGBoost : {'✅' if XGB_AVAILABLE else '❌'}")
st.sidebar.markdown(f"- LightGBM : {'✅' if LGBM_AVAILABLE else '❌'}")
st.sidebar.markdown(f"- SHAP : {'✅' if SHAP_AVAILABLE else '❌'}")
st.sidebar.markdown(f"- joblib : {'✅' if JOBLIB_AVAILABLE else '❌'}")

st.sidebar.markdown("---")
st.sidebar.markdown("**Slimane Rahim**")
st.sidebar.markdown("Bioinformaticien | Génomique")


# ============================================================
# PAGE 1 : ACCUEIL
# ============================================================
if page == "🏠 Accueil":
    st.markdown('<h1 class="main-header">🧬 BioMission Suite</h1>', unsafe_allow_html=True)
    st.markdown("### Plateforme intégrée de bio-informatique et d'analyse génomique")
    st.caption("Version 2.0 — Pipeline ML avancé")

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Modules", "8")
    col2.metric("Formats supportés", "10+")
    col3.metric("Analyses", "15+")
    col4.metric("Statut", "✅ Opérationnel")

    st.markdown("---")
    st.markdown("### 📋 Modules disponibles")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **📤 Upload de données**
        - FASTQ, VCF, PED/MAP
        - POD5, FAST5 (Nanopore)
        - CSV, TSV

        **🔍 Contrôle Qualité**
        - Statistiques FASTQ
        - Distribution des longueurs
        - Contenu GC

        **🧬 PLINK**
        - Lecture PED/MAP
        - Génotypes et variants
        - Statistiques de base

        **🔬 Nanopore**
        - Lecture POD5 (signal brut)
        - Lecture FAST5
        - FASTQ basecalled
        """)

    with c2:
        st.markdown("""
        **📊 Variants VCF**
        - Parsing VCF
        - Statistiques de variants
        - Types (SNP/INDEL)

        **🤖 Machine Learning avancé**
        - Pipeline complet (ColumnTransformer)
        - Validation croisée k-fold
        - GridSearchCV
        - XGBoost / LightGBM
        - SHAP interpretabilité
        - Export .pkl + rapport HTML

        **📄 Rapports**
        - Export Markdown / HTML

        **🔗 Extensions futures**
        - API REST (Hetzner)
        - Pipelines Nextflow
        """)

    st.markdown("---")
    st.info("💡 **Astuce** : Utilisez le menu latéral pour naviguer entre les modules.")


# ============================================================
# PAGE 2 : UPLOAD
# ============================================================
elif page == "📤 Upload":
    st.header("📤 Upload de données")
    st.markdown("Chargez vos fichiers pour analyse. Les fichiers restent en mémoire.")

    uploaded = st.file_uploader(
        "Choisissez un ou plusieurs fichiers",
        type=["fastq", "fq", "gz", "vcf", "ped", "map", "pod5", "fast5", "csv", "tsv", "txt"],
        accept_multiple_files=True
    )

    if uploaded:
        st.success(f"✅ {len(uploaded)} fichier(s) chargé(s)")

        data = []
        for f in uploaded:
            size_mb = f.size / (1024 * 1024)
            data.append({
                "Nom": f.name,
                "Taille": f"{size_mb:.2f} MB",
                "Type": f.type or "inconnu"
            })

        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        st.info("💡 Naviguez vers le module approprié dans le menu latéral.")


# ============================================================
# PAGE 3 : QC FASTQ
# ============================================================
elif page == "🔍 QC FASTQ":
    st.header("🔍 Contrôle Qualité FASTQ")

    uploaded = st.file_uploader("Fichier FASTQ ou FASTQ.gz", type=["fastq", "fq", "gz"])
    max_reads = st.slider("Nombre max de reads à analyser", 1000, 100000, 10000, 1000)

    if uploaded and st.button("Lancer l'analyse QC", type="primary"):
        with st.spinner("Lecture du fichier..."):
            try:
                reads = read_fastq_file(uploaded, max_reads=max_reads)
                st.success(f"✅ {len(reads)} reads analysés")

                if len(reads) == 0:
                    st.warning("Aucun read détecté. Vérifiez le format.")
                    st.stop()

                df = pd.DataFrame([{
                    "read_id": r["read_id"],
                    "length": r["length"],
                    "gc_content": sum(1 for c in r["sequence"].upper() if c in "GC") / max(r["length"], 1) * 100
                } for r in reads])

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Reads totaux", f"{len(df):,}")
                col2.metric("Longueur moyenne", f"{df['length'].mean():.1f} bp")
                col3.metric("Longueur médiane", f"{df['length'].median():.0f} bp")
                col4.metric("GC moyen", f"{df['gc_content'].mean():.1f} %")

                st.markdown("---")

                c1, c2 = st.columns(2)
                with c1:
                    fig = px.histogram(df, x="length", nbins=50,
                                       title="Distribution des longueurs",
                                       color_discrete_sequence=["#667eea"])
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    fig = px.histogram(df, x="gc_content", nbins=50,
                                       title="Distribution du contenu GC",
                                       color_discrete_sequence=["#764ba2"])
                    st.plotly_chart(fig, use_container_width=True)

                with st.expander("📋 Voir les statistiques détaillées"):
                    st.dataframe(df.describe(), use_container_width=True)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Télécharger les statistiques (CSV)", csv, "qc_stats.csv")

            except Exception as e:
                st.error(f"❌ Erreur : {str(e)}")


# ============================================================
# PAGE 4 : PLINK PED/MAP
# ============================================================
elif page == "🧬 PLINK (PED/MAP)":
    st.header("🧬 Analyse PLINK (PED/MAP)")
    st.markdown("Chargez un fichier `.ped` et son `.map` correspondant.")

    col1, col2 = st.columns(2)
    with col1:
        ped_file = st.file_uploader("Fichier .ped", type=["ped", "txt"])
    with col2:
        map_file = st.file_uploader("Fichier .map", type=["map", "txt"])

    if ped_file and map_file and st.button("Analyser PLINK", type="primary"):
        with st.spinner("Lecture des fichiers..."):
            try:
                ped_content = ped_file.read().decode("utf-8", errors="ignore")
                map_content = map_file.read().decode("utf-8", errors="ignore")

                samples, variants, genotypes = read_plink_ped_map(ped_content, map_content)

                st.success(f"✅ {len(samples)} échantillons, {len(variants)} variants chargés")

                col1, col2, col3 = st.columns(3)
                col1.metric("Échantillons", len(samples))
                col2.metric("Variants", len(variants))
                col3.metric("Colonnes génotypes", genotypes.shape[1])

                st.markdown("---")

                tab1, tab2, tab3, tab4 = st.tabs(["👥 Échantillons", "🧬 Variants", "📊 Génotypes", "📈 Statistiques"])

                with tab1:
                    st.dataframe(samples, use_container_width=True)
                    csv = samples.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Télécharger Échantillons", csv, "samples.csv")

                with tab2:
                    st.dataframe(variants, use_container_width=True)
                    csv = variants.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Télécharger Variants", csv, "variants.csv")

                with tab3:
                    st.dataframe(genotypes.head(20), use_container_width=True)

                with tab4:
                    chrom_counts = variants["chrom"].astype(str).value_counts().reset_index()
                    chrom_counts.columns = ["Chromosome", "Nombre"]
                    fig = px.bar(chrom_counts, x="Chromosome", y="Nombre",
                                 title="Variants par chromosome",
                                 color_discrete_sequence=["#667eea"])
                    st.plotly_chart(fig, use_container_width=True)

                    fig = px.histogram(variants, x="pos", nbins=50,
                                       title="Distribution des positions",
                                       color_discrete_sequence=["#764ba2"])
                    st.plotly_chart(fig, use_container_width=True)

                    if "sex" in samples.columns:
                        sex_counts = samples["sex"].value_counts().reset_index()
                        sex_counts.columns = ["Sexe", "Nombre"]
                        fig = px.pie(sex_counts, names="Sexe", values="Nombre",
                                     title="Répartition par sexe")
                        st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Erreur : {str(e)}")
                st.info("Vérifiez que les fichiers sont au format PLINK standard.")


# ============================================================
# PAGE 5 : NANOPORE
# ============================================================
elif page == "🔬 Nanopore (POD5/FAST5)":
    st.header("🔬 Données Nanopore")

    tab1, tab2, tab3 = st.tabs(["📦 POD5 (signal brut)", "📦 FAST5 (ancien format)", "📄 FASTQ (basecalled)"])

    with tab1:
        st.subheader("POD5 - Signal brut")
        st.markdown("Format moderne de Nanopore contenant les signaux bruts.")

        if not POD5_AVAILABLE:
            st.warning("⚠️ La bibliothèque `pod5` n'est pas installée.")
            st.info("Ajoutez `pod5` dans requirements.txt. La lecture POD5 est gourmande en mémoire, préférez l'exécution sur un serveur dédié (Hetzner) pour les gros fichiers.")
        else:
            pod5_file = st.file_uploader("Fichier POD5", type=["pod5"])
            max_reads = st.slider("Nombre max de reads", 10, 1000, 100, 10, key="pod5_max")

            if pod5_file and st.button("Analyser POD5", type="primary"):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pod5") as tmp:
                        tmp.write(pod5_file.getbuffer())
                        tmp_path = tmp.name

                    with st.spinner("Lecture des signaux..."):
                        reads = []
                        with pod5.Reader(tmp_path) as reader:
                            for i, read in enumerate(reader.reads()):
                                if i >= max_reads:
                                    break
                                reads.append({
                                    "read_id": str(read.read_id),
                                    "num_samples": len(read.signal),
                                    "channel": getattr(read, "channel", None),
                                    "sample_rate": getattr(read, "sample_rate", None)
                                })

                    st.success(f"✅ {len(reads)} reads lus")
                    df = pd.DataFrame(reads)
                    st.dataframe(df, use_container_width=True)

                    fig = px.histogram(df, x="num_samples", nbins=50,
                                       title="Distribution des longueurs de signal")
                    st.plotly_chart(fig, use_container_width=True)

                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Télécharger CSV", csv, "pod5_stats.csv")
                except Exception as e:
                    st.error(f"❌ Erreur : {str(e)}")
                finally:
                    if 'tmp_path' in locals() and os.path.exists(tmp_path):
                        os.unlink(tmp_path)

    with tab2:
        st.subheader("FAST5 - Ancien format")
        st.markdown("Format historique de Nanopore. Utilisez POD5 si possible.")

        fast5_file = st.file_uploader("Fichier FAST5", type=["fast5"])

        if fast5_file and st.button("Analyser FAST5", type="primary"):
            try:
                try:
                    from ont_fast5_api.fast5_interface import get_fast5_file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".fast5") as tmp:
                        tmp.write(fast5_file.getbuffer())
                        tmp_path = tmp.name

                    reads = []
                    with get_fast5_file(tmp_path, mode="r") as f5:
                        for i, read in enumerate(f5.get_reads()):
                            if i >= 100:
                                break
                            signal = read.get_raw_data()
                            reads.append({
                                "read_id": read.read_id,
                                "num_samples": len(signal)
                            })

                    df = pd.DataFrame(reads)
                    st.success(f"✅ {len(df)} reads lus")
                    st.dataframe(df)
                except ImportError:
                    st.error("❌ `ont-fast5-api` non installé. Ajoutez-le à requirements.txt.")
            except Exception as e:
                st.error(f"❌ Erreur : {str(e)}")
            finally:
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

    with tab3:
        st.subheader("FASTQ Nanopore (basecalled)")
        st.markdown("Reads déjà convertis en séquences par le basecaller.")

        fastq_file = st.file_uploader("Fichier FASTQ ou FASTQ.gz", type=["fastq", "fq", "gz"], key="nanopore_fastq")
        max_reads = st.slider("Nombre max de reads", 100, 50000, 5000, 100, key="np_fastq_max")

        if fastq_file and st.button("Analyser FASTQ Nanopore", type="primary"):
            with st.spinner("Analyse..."):
                try:
                    reads = read_fastq_file(fastq_file, max_reads=max_reads)
                    st.success(f"✅ {len(reads)} reads analysés")

                    df = pd.DataFrame([{
                        "read_id": r["read_id"],
                        "length": r["length"]
                    } for r in reads])

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Reads", f"{len(df):,}")
                    col2.metric("Longueur moyenne", f"{df['length'].mean():.0f} bp")
                    col3.metric("Longueur max", f"{df['length'].max():,} bp")

                    fig = px.histogram(df, x="length", nbins=100,
                                       title="Distribution des longueurs (échelle log)",
                                       color_discrete_sequence=["#667eea"])
                    fig.update_yaxes(type="log")
                    st.plotly_chart(fig, use_container_width=True)

                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Télécharger CSV", csv, "nanopore_fastq.csv")
                except Exception as e:
                    st.error(f"❌ Erreur : {str(e)}")


# ============================================================
# PAGE 6 : VCF
# ============================================================
elif page == "📊 VCF Variants":
    st.header("📊 Analyse de variants VCF")

    uploaded = st.file_uploader("Fichier VCF ou VCF.gz", type=["vcf", "gz"])
    max_variants = st.slider("Nombre max de variants", 1000, 500000, 50000, 1000)

    if uploaded and st.button("Analyser VCF", type="primary"):
        with st.spinner("Parsing du VCF..."):
            try:
                raw = uploaded.read()
                if uploaded.name.endswith(".gz"):
                    content = gzip.decompress(raw).decode("utf-8", errors="ignore")
                else:
                    content = raw.decode("utf-8", errors="ignore")

                variants = parse_vcf_simple(content, max_variants=max_variants)

                if len(variants) == 0:
                    st.warning("Aucun variant trouvé.")
                    st.stop()

                df = pd.DataFrame(variants)
                st.success(f"✅ {len(df):,} variants analysés")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Variants", f"{len(df):,}")
                col2.metric("Chromosomes", df["CHROM"].nunique())
                col3.metric("Qualité moyenne", f"{df['QUAL'].mean():.1f}")
                col4.metric("SNPs", f"{len(df[df['REF'].str.len() == 1]):,}")

                st.markdown("---")

                tab1, tab2, tab3 = st.tabs(["📋 Données", "📈 Distribution", "🧬 Types"])

                with tab1:
                    st.dataframe(df.head(1000), use_container_width=True)
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Télécharger CSV", csv, "variants.csv")

                with tab2:
                    chrom_counts = df["CHROM"].astype(str).value_counts().head(25).reset_index()
                    chrom_counts.columns = ["Chromosome", "Nombre"]
                    fig = px.bar(chrom_counts, x="Chromosome", y="Nombre",
                                 title="Variants par chromosome",
                                 color_discrete_sequence=["#667eea"])
                    st.plotly_chart(fig, use_container_width=True)

                    fig = px.histogram(df, x="QUAL", nbins=50,
                                       title="Distribution des scores de qualité",
                                       color_discrete_sequence=["#764ba2"])
                    st.plotly_chart(fig, use_container_width=True)

                with tab3:
                    df["type"] = df.apply(
                        lambda r: "SNP" if len(r["REF"]) == 1 and len(r["ALT"]) == 1
                        else "INDEL" if len(r["REF"]) != len(r["ALT"])
                        else "Autre", axis=1
                    )
                    type_counts = df["type"].value_counts().reset_index()
                    type_counts.columns = ["Type", "Nombre"]
                    fig = px.pie(type_counts, names="Type", values="Nombre",
                                 title="Types de variants")
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Erreur : {str(e)}")


# ============================================================
# PAGE 7 : MACHINE LEARNING (v2 — production-ready)
# ============================================================
elif page == "🤖 Machine Learning":
    st.header("🤖 Machine Learning — Pipeline avancé")

    st.warning(
        "⚠️ **Ce module est un outil d'exploration, pas un dispositif médical.** "
        "Les prédictions ne doivent pas être utilisées pour un diagnostic clinique. "
        "Vérifiez toujours avec validation croisée et un expert du domaine."
    )

    if not SKLEARN_AVAILABLE:
        st.error("❌ scikit-learn n'est pas installé. Ajoutez-le à requirements.txt.")
        st.stop()

    uploaded = st.file_uploader("Fichier CSV avec données et variable cible", type=["csv"])

    if not uploaded:
        st.info("💡 Chargez un CSV : 1 ligne = 1 échantillon, 1 colonne = 1 variable.")
        st.stop()

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"❌ Erreur de lecture CSV : {e}")
        st.stop()

    st.success(f"✅ {df.shape[0]} lignes, {df.shape[1]} colonnes")
    with st.expander("👀 Aperçu des données"):
        st.dataframe(df.head(20), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        target = st.selectbox("🎯 Variable cible", df.columns,
                              help="Ce que vous voulez prédire.")
    with col_b:
        features = st.multiselect(
            "📊 Variables explicatives",
            [c for c in df.columns if c != target],
            default=[c for c in df.columns if c != target][:5],
            help="Colonnes numériques ET catégorielles acceptées."
        )

    if not features:
        st.warning("Sélectionnez au moins une variable explicative.")
        st.stop()

    X = df[features].copy()
    y_raw = df[target].copy()

    is_numeric_target = pd.api.types.is_numeric_dtype(y_raw)
    n_unique = y_raw.nunique()

    if is_numeric_target and n_unique > 15:
        task_type = "regression"
        st.info(f"🔍 Cible continue détectée ({n_unique} valeurs uniques) → **Régression**")
    else:
        task_type = "classification"
        st.info(f"🔍 Cible catégorielle détectée ({n_unique} classes) → **Classification**")

    with st.expander("🔎 Diagnostic des données", expanded=True):
        diag_col1, diag_col2, diag_col3 = st.columns(3)

        n_missing_X = X.isna().sum().sum()
        n_missing_y = y_raw.isna().sum()
        diag_col1.metric("NaN dans X", f"{n_missing_X}")
        diag_col2.metric("NaN dans y", f"{n_missing_y}")

        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        diag_col3.metric("Colonnes num./cat.", f"{len(num_cols)}/{len(cat_cols)}")

        if task_type == "classification" and n_unique == 2:
            counts = y_raw.value_counts()
            ratio = counts.min() / counts.max()
            if ratio < 0.3:
                st.warning(
                    f"⚠️ **Déséquilibre de classes détecté** "
                    f"(ratio minoritaire/majoritaire = {ratio:.2f}). "
                    "`class_weight='balanced'` sera appliqué automatiquement."
                )

    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Taille du test", 0.1, 0.4, 0.2, 0.05)
    with col2:
        n_folds = st.slider("Folds validation croisée", 3, 10, 5, 1)

    model_choices = ["Random Forest", "Gradient Boosting", "Régression Logistique"]
    if XGB_AVAILABLE:
        model_choices.append("XGBoost")
    if LGBM_AVAILABLE:
        model_choices.append("LightGBM")

    selected_models = st.multiselect(
        "🤖 Modèles à comparer",
        model_choices,
        default=["Random Forest"] + (["XGBoost"] if XGB_AVAILABLE else [])
    )

    if not selected_models:
        st.warning("Sélectionnez au moins un modèle.")
        st.stop()

    if st.button("🚀 Entraîner et comparer", type="primary"):
        with st.spinner("Construction du pipeline et entraînement..."):

            num_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
            cat_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])

            preprocessor = ColumnTransformer([
                ("num", num_pipe, num_cols),
                ("cat", cat_pipe, cat_cols)
            ], remainder="drop")

            if task_type == "classification":
                le = LabelEncoder()
                mask = y_raw.notna()
                X_clean = X[mask]
                y = le.fit_transform(y_raw[mask].astype(str))
                class_names = le.classes_
            else:
                mask = y_raw.notna()
                X_clean = X[mask]
                y = y_raw[mask].values.astype(float)
                class_names = None

            class_weight = "balanced" if (
                task_type == "classification" and n_unique == 2
                and (y_raw.value_counts().min() / y_raw.value_counts().max()) < 0.3
            ) else None

            model_map = {}
            for name in selected_models:
                if task_type == "classification":
                    if name == "Random Forest":
                        model_map[name] = RandomForestClassifier(
                            n_estimators=200, class_weight=class_weight,
                            random_state=42, n_jobs=-1)
                    elif name == "Gradient Boosting":
                        model_map[name] = GradientBoostingClassifier(random_state=42)
                    elif name == "Régression Logistique":
                        model_map[name] = LogisticRegression(
                            max_iter=2000, class_weight=class_weight)
                    elif name == "XGBoost" and XGB_AVAILABLE:
                        model_map[name] = xgb.XGBClassifier(
                            n_estimators=200, eval_metric="logloss",
                            random_state=42, n_jobs=-1)
                    elif name == "LightGBM" and LGBM_AVAILABLE:
                        model_map[name] = lgb.LGBMClassifier(
                            n_estimators=200, random_state=42, n_jobs=-1,
                            class_weight=class_weight, verbose=-1)
                else:
                    if name == "Random Forest":
                        model_map[name] = RandomForestRegressor(
                            n_estimators=200, random_state=42, n_jobs=-1)
                    elif name == "Gradient Boosting":
                        model_map[name] = GradientBoostingRegressor(random_state=42)
                    elif name == "Régression Logistique":
                        model_map[name] = Ridge()
                    elif name == "XGBoost" and XGB_AVAILABLE:
                        model_map[name] = xgb.XGBRegressor(
                            n_estimators=200, random_state=42, n_jobs=-1)
                    elif name == "LightGBM" and LGBM_AVAILABLE:
                        model_map[name] = lgb.LGBMRegressor(
                            n_estimators=200, random_state=42, n_jobs=-1, verbose=-1)

            cv = (StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
                  if task_type == "classification"
                  else KFold(n_splits=n_folds, shuffle=True, random_state=42))

            scoring = (["roc_auc", "f1_weighted", "accuracy"]
                       if task_type == "classification"
                       else ["r2", "neg_mean_squared_error", "neg_mean_absolute_error"])

            cv_results = []
            progress = st.progress(0)
            for i, (name, clf) in enumerate(model_map.items()):
                pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
                scores = cross_validate(pipe, X_clean, y, cv=cv, scoring=scoring,
                                        n_jobs=-1, return_train_score=False)

                if task_type == "classification":
                    cv_results.append({
                        "Modèle": name,
                        "AUC moyen": f"{scores['test_roc_auc'].mean():.4f}",
                        "AUC ±": f"± {scores['test_roc_auc'].std():.4f}",
                        "F1 moyen": f"{scores['test_f1_weighted'].mean():.4f}",
                        "Accuracy": f"{scores['test_accuracy'].mean():.4f}"
                    })
                else:
                    cv_results.append({
                        "Modèle": name,
                        "R² moyen": f"{scores['test_r2'].mean():.4f}",
                        "R² ±": f"± {scores['test_r2'].std():.4f}",
                        "RMSE": f"{np.sqrt(-scores['test_neg_mean_squared_error'].mean()):.4f}",
                        "MAE": f"{-scores['test_neg_mean_absolute_error'].mean():.4f}"
                    })
                progress.progress((i + 1) / len(model_map))

            st.markdown("### 📊 Comparaison des modèles (validation croisée)")
            cv_df = pd.DataFrame(cv_results)
            st.dataframe(cv_df, use_container_width=True)

            if task_type == "classification":
                best_name = max(
                    model_map.keys(),
                    key=lambda n: float(
                        cv_df.loc[cv_df["Modèle"] == n, "AUC moyen"].values[0]
                    )
                )
            else:
                best_name = max(
                    model_map.keys(),
                    key=lambda n: float(
                        cv_df.loc[cv_df["Modèle"] == n, "R² moyen"].values[0]
                    )
                )

            st.success(f"🏆 Meilleur modèle : **{best_name}**")

            with st.expander("⚙️ Tuning des hyperparamètres (GridSearchCV)"):
                tune = st.checkbox("Lancer le tuning (peut prendre du temps)", value=False)

            best_pipe = Pipeline([("prep", preprocessor), ("clf", model_map[best_name])])

            if tune:
                with st.spinner(f"Tuning de {best_name}..."):
                    if task_type == "classification":
                        param_grid = {
                            "clf__n_estimators": [100, 300],
                            "clf__max_depth": [None, 10, 20]
                        } if best_name in ("Random Forest", "XGBoost", "LightGBM") else {
                            "clf__C": [0.1, 1.0, 10.0]
                        }
                    else:
                        param_grid = {
                            "clf__n_estimators": [100, 300],
                            "clf__max_depth": [None, 10, 20]
                        } if best_name in ("Random Forest", "XGBoost", "LightGBM") else {
                            "clf__alpha": [0.1, 1.0, 10.0]
                        }

                    scoring_tune = "roc_auc" if task_type == "classification" else "r2"
                    grid = GridSearchCV(best_pipe, param_grid, cv=cv,
                                        scoring=scoring_tune, n_jobs=-1)
                    grid.fit(X_clean, y)
                    best_pipe = grid.best_estimator_
                    st.success(f"✅ Meilleurs paramètres : `{grid.best_params_}`")

            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y, test_size=test_size,
                random_state=42,
                stratify=y if task_type == "classification" else None
            )
            best_pipe.fit(X_train, y_train)
            y_pred = best_pipe.predict(X_test)

            st.markdown("### 🎯 Performance sur le jeu de test")

            if task_type == "classification":
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📋 Rapport de classification")
                    report = classification_report(
                        y_test, y_pred,
                        target_names=[str(c) for c in class_names] if class_names is not None else None,
                        output_dict=True
                    )
                    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

                with col2:
                    st.subheader("🧮 Matrice de confusion")
                    cm = confusion_matrix(y_test, y_pred)
                    fig = px.imshow(
                        cm, text_auto=True, aspect="auto",
                        labels=dict(x="Prédit", y="Réel", color="Nombre"),
                        x=[str(c) for c in class_names] if class_names is not None else None,
                        y=[str(c) for c in class_names] if class_names is not None else None,
                        color_continuous_scale="Blues"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                if n_unique == 2:
                    y_prob = best_pipe.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    roc_auc = auc(fpr, tpr)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=fpr, y=tpr,
                                             name=f"AUC = {roc_auc:.3f}",
                                             line=dict(color="#764ba2", width=3)))
                    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                                             name="Aléatoire",
                                             line=dict(dash="dash", color="gray")))
                    fig.update_layout(title="Courbe ROC",
                                      xaxis_title="Taux de faux positifs",
                                      yaxis_title="Taux de vrais positifs")
                    st.plotly_chart(fig, use_container_width=True)

            else:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("R²", f"{r2_score(y_test, y_pred):.4f}")
                col2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
                col3.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.4f}")
                col4.metric("Échantillons test", f"{len(y_test)}")

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers",
                                         marker=dict(color="#667eea", opacity=0.6)))
                fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
                                         y=[y_test.min(), y_test.max()],
                                         mode="lines", name="Idéal",
                                         line=dict(dash="dash", color="gray")))
                fig.update_layout(title="Prédit vs Réel",
                                  xaxis_title="Réel", yaxis_title="Prédit")
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 📈 Importance des variables")
            try:
                with st.spinner("Calcul de l'importance (permutation)..."):
                    perm = permutation_importance(
                        best_pipe, X_test, y_test,
                        n_repeats=10, random_state=42, n_jobs=-1,
                        scoring="roc_auc" if task_type == "classification" else "r2"
                    )
                    imp_df = pd.DataFrame({
                        "feature": X.columns,
                        "importance": perm.importances_mean,
                        "std": perm.importances_std
                    }).sort_values("importance", ascending=False).head(20)

                    fig = px.bar(imp_df, x="importance", y="feature",
                                 orientation="h", error_x="std",
                                 color_discrete_sequence=["#667eea"],
                                 title="Permutation Importance (top 20)")
                    fig.update_layout(yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.info(f"Importance par permutation indisponible : {e}")

            if SHAP_AVAILABLE and task_type == "classification":
                with st.expander("🔬 Interprétabilité SHAP"):
                    try:
                        X_test_transformed = best_pipe.named_steps["prep"].transform(X_test)
                        try:
                            feat_names = (best_pipe.named_steps["prep"]
                                          .get_feature_names_out())
                        except Exception:
                            feat_names = [f"f{i}" for i in range(X_test_transformed.shape[1])]

                        clf = best_pipe.named_steps["clf"]
                        if hasattr(clf, "feature_importances_"):
                            explainer = shap.TreeExplainer(clf)
                            shap_values = explainer.shap_values(X_test_transformed)
                            if isinstance(shap_values, list):
                                shap_values = shap_values[0]
                        else:
                            explainer = shap.LinearExplainer(clf, X_test_transformed)
                            shap_values = explainer.shap_values(X_test_transformed)

                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(figsize=(10, 6))
                        shap.summary_plot(shap_values, X_test_transformed,
                                          feature_names=feat_names, show=False,
                                          plot_type="bar")
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as e:
                        st.warning(f"SHAP indisponible pour ce modèle : {e}")

            if JOBLIB_AVAILABLE:
                st.markdown("### 💾 Sauvegarde du modèle")
                col1, col2 = st.columns(2)
                with col1:
                    model_name = st.text_input("Nom du fichier", "biomission_model")
                    if st.button("📦 Préparer le .pkl"):
                        buffer = BytesIO()
                        joblib.dump(best_pipe, buffer)
                        buffer.seek(0)
                        st.download_button(
                            "📥 Télécharger le modèle (.pkl)",
                            buffer.getvalue(),
                            f"{model_name}.pkl",
                            "application/octet-stream"
                        )
                with col2:
                    st.caption("Le `.pkl` contient **tout le pipeline** "
                               "(preprocessing + modèle). Rechargeable via "
                               "`joblib.load('model.pkl')`.")

            st.markdown("### 📄 Rapport HTML")
            if st.button("🧾 Générer le rapport"):
                html = f"""
                <html><head><meta charset="utf-8">
                <style>
                  body {{ font-family: Arial, sans-serif; margin: 40px; color:#222; }}
                  h1 {{ color:#1f4e79; }}
                  h2 {{ color:#667eea; border-bottom:2px solid #667eea; }}
                  table {{ border-collapse: collapse; width:100%; margin:10px 0; }}
                  th, td {{ border:1px solid #ddd; padding:8px; text-align:left; }}
                  th {{ background:#f0f4ff; }}
                  .warn {{ background:#fff4e5; padding:12px; border-left:4px solid #ff9800; }}
                </style></head><body>
                <h1>🧬 Rapport BioMission Suite — Machine Learning</h1>
                <p><b>Date :</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
                <p><b>Fichier :</b> {uploaded.name}</p>
                <div class="warn">
                  ⚠️ Outil d'exploration uniquement — ne pas utiliser pour un diagnostic clinique.
                </div>
                <h2>Données</h2>
                <ul>
                  <li>Lignes : {df.shape[0]}</li>
                  <li>Colonnes : {df.shape[1]}</li>
                  <li>Type de tâche : {task_type}</li>
                  <li>Cible : {target}</li>
                  <li>Features : {', '.join(features)}</li>
                  <li>NaN dans X : {n_missing_X}</li>
                </ul>
                <h2>Comparaison des modèles (CV {n_folds}-fold)</h2>
                {cv_df.to_html(index=False)}
                <h2>Meilleur modèle : {best_name}</h2>
                <h2>Performance test (hold-out {int(test_size*100)}%)</h2>
                <pre>{classification_report(y_test, y_pred) if task_type == 'classification' else f'R2={r2_score(y_test, y_pred):.4f}, RMSE={np.sqrt(mean_squared_error(y_test, y_pred)):.4f}'}</pre>
                <hr>
                <p style="color:gray;font-size:0.85em;">
                  Généré par BioMission Suite v2.0 — Slimane Rahim
                </p>
                </body></html>
                """
                st.download_button(
                    "📥 Télécharger le rapport (.html)",
                    html.encode("utf-8"),
                    "rapport_ml.html",
                    "text/html"
                )

    st.markdown("---")
    st.caption(
        "⚠️ Ce module ne remplace pas une validation biologique. "
        "Toute conclusion doit être confirmée expérimentalement."
    )


# ============================================================
# PAGE 8 : RAPPORTS
# ============================================================
elif page == "📄 Rapports":
    st.header("📄 Rapports et exports")

    st.markdown("### Générer un rapport d'analyse")

    col1, col2 = st.columns(2)
    with col1:
        title = st.text_input("Titre du rapport", "Rapport BioMission")
        author = st.text_input("Auteur", "Slimane Rahim")
    with col2:
        module = st.selectbox("Module analysé",
                              ["QC FASTQ", "PLINK", "Nanopore", "VCF", "ML"])
        notes = st.text_area("Notes", "Analyse réalisée le " + datetime.now().strftime("%Y-%m-%d"))

    if st.button("Générer le rapport", type="primary"):
        report = f"""# {title}

**Auteur :** {author}
**Date :** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Module :** {module}

## Notes

{notes}

---

*Rapport généré automatiquement par BioMission Suite*
"""
        st.markdown(report)

        st.download_button(
            "📥 Télécharger le rapport (Markdown)",
            report.encode("utf-8"),
            "rapport.md",
            "text/markdown"
        )


# ============================================================
# FOOTER
# ============================================================
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.8rem;'>"
    "BioMission Suite v2.0<br>© 2026 Slimane Rahim"
    "</div>",
    unsafe_allow_html=True
)
