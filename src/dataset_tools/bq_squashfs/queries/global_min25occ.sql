-- Global dataset: all downloaded images for species with >= 25 distinct GBIF occurrences.
-- Expected: ~4,704 species, ~10.6M images.
WITH qualifying_species AS (
    SELECT tx.species_name
    FROM `leps-ai.global_butterflies_2604.training_images` ti
    JOIN `leps-ai.global_butterflies_2604.inat_taxa` tx USING (inat_taxon_id)
    WHERE ti.fetch_status = 'downloaded'
      AND tx.species_name IS NOT NULL
    GROUP BY tx.species_name
    HAVING COUNT(DISTINCT ti.gbif_id) >= 25
)
SELECT
    ti.photo_id,
    ti.gbif_id,
    ti.relative_local_path,
    ti.dataset_source_uuid,
    ti.inat_taxon_id,
    tx.species_name,
    tx.family
FROM `leps-ai.global_butterflies_2604.training_images` ti
JOIN `leps-ai.global_butterflies_2604.inat_taxa` tx USING (inat_taxon_id)
WHERE ti.fetch_status = 'downloaded'
  AND tx.species_name IN (SELECT species_name FROM qualifying_species)
