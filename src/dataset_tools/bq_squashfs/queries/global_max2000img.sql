-- Global dataset: all downloaded images, capped at 2000 images per species.
-- Species with fewer than 2000 images are included in full.
-- Expected: ~12,313 species, ~2.7M images.
WITH ranked AS (
    SELECT
        ti.photo_id,
        ti.gbif_id,
        ti.relative_local_path,
        ti.dataset_source_uuid,
        ti.inat_taxon_id,
        tx.species_name,
        tx.family,
        ROW_NUMBER() OVER (
            PARTITION BY tx.species_name
            ORDER BY ti.photo_id
        ) AS rn
    FROM `leps-ai.global_butterflies_2604.training_images` ti
    JOIN `leps-ai.global_butterflies_2604.inat_taxa` tx USING (inat_taxon_id)
    WHERE ti.fetch_status = 'downloaded'
      AND tx.species_name IS NOT NULL
)
SELECT
    photo_id,
    gbif_id,
    relative_local_path,
    dataset_source_uuid,
    inat_taxon_id,
    species_name,
    family
FROM ranked
WHERE rn <= 2000
