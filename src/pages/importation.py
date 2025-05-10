import streamlit as st
import zipfile
import json
import pymongo


class ImportPage:

    @staticmethod
    def page_import(
        collection_regular_paper: pymongo.collection.Collection,
        collection_dataset_paper: pymongo.collection.Collection,
    ):
        st.title("Import JSON Data for Papers")
        select_paper_type_cols = st.columns([1, 1, 1, 0.8])
        with select_paper_type_cols[0]:
            paper_type = st.selectbox(
                "Select Paper Type", ["", "Regular Paper", "Dataset Paper"]
            )
        if paper_type:
            # Upload ZIP file
            uploaded_file = st.file_uploader(
                "Upload a ZIP file containing JSON files", type=["zip"]
            )

            if uploaded_file is not None:
                with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
                    json_files = [f for f in zip_ref.namelist() if f.endswith(".json")]

                    if not json_files:
                        st.error("[Error] No JSON files found in the ZIP!")
                    else:
                        st.success(
                            f"[Success] Found {len(json_files)} JSON files, processing..."
                        )
                        result_message = st.empty()
                        progress_bar = st.progress(0)
                        new_data = []
                        for i, json_filename in enumerate(json_files):
                            with zip_ref.open(json_filename) as json_file:
                                try:
                                    data = json.load(json_file)
                                    # Filter out empty JSON or `{}`
                                    if data and data != {}:
                                        new_data.append({"content": data})
                                        st.success(
                                            f"[Success] Processing `{json_filename}`."
                                        )
                                    else:
                                        st.warning(
                                            f"[Skipped]`{json_filename}` is empty."
                                        )

                                except json.JSONDecodeError:
                                    st.warning(
                                        f"[Skipped]`{json_filename}` is not a valid JSON file."
                                    )

                            # Update progress bar
                            progress_bar.progress((i + 1) / len(json_files))

                        # Insert valid JSON into MongoDB
                        if new_data:
                            if paper_type == "Regular Paper":
                                collection_regular_paper.insert_many(new_data)
                            elif paper_type == "Dataset Paper":
                                collection_dataset_paper.insert_many(new_data)
                            result_message.success(
                                f"[Success] Successfully imported {len(new_data)} JSON files into MongoDB."
                            )
                        else:
                            result_message.error(
                                "[Error] No valid JSON data found for import."
                            )
