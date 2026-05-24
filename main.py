"""
main.py — Orquestador principal del pipeline.
Ejecutar con:
python main.py
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.train import entrenar

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

log = logging.getLogger("MAIN")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data',
        default='data/2_DS_train_enf_corazon.csv',
        help='Ruta del dataset'
    )

    parser.add_argument(
        '--artifact-dir',
        default='artifacts',
        help='Directorio de artifacts'
    )

    args = parser.parse_args()

    log.info('Iniciando pipeline principal')

    entrenar(
        ruta_data=args.data,
        artifact_dir=args.artifact_dir
    )

    log.info('Pipeline finalizado correctamente')


if __name__ == '__main__':
    main()