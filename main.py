from mapper import AutoMapper
from utils import parse_csv_headers_local, parse_target_format
import json


def main():
    # Initialize the matcher
    mapper = AutoMapper()

    # Example field lists
    # target_fields = ["mailing_line_1", "mailing_line_2", "date_of_birth", "telephone", "email_address", "first_name", "last_name", "primary_taxonomy"]

    with open("common_format.json") as target_format_file:
        data = json.load(target_format_file)
        target_fields, stripped_config = parse_target_format(data)

    # Create example pairs for optimization
    # examples = [
    #     ("address_1", "mailing_line_1", target_fields),
    #     ("address_2", "mailing_line_2", target_fields),
    #     ("DOB", "date_of_birth", target_fields),
    #     ("phone_num", "telephone", target_fields),
    #     ("email", "email_address", target_fields),
    #     ("fname", "first_name", target_fields),
    #     ("lname", "last_name", target_fields),
    #     # Additional examples for different formats
    #     ("street1", "mailing_line_1", target_fields),
    #     ("street2", "mailing_line_2", target_fields),
    #     ("birth_date", "date_of_birth", target_fields),
    #     ("tel", "telephone", target_fields),
    #     ("e_mail", "email_address", target_fields),
    #     ("first_name", "first_name", target_fields),
    #     ("last_name", "last_name", target_fields),
    # ]

    # Optimize the matchers with examples
    # matcher.optimize_with_examples(examples)

    # test_sources = ["addr1", "second_addr", "birthday", "contact_phone", "fname", "nucc_code"]
    test_sources = parse_csv_headers_local(
        "defactoorderly_sample_first_507_records.csv"
    )
    batch_results = mapper.batch_match_fields(test_sources, target_fields)
    updated_config = mapper.update_config_with_matches(stripped_config, batch_results)
    print(updated_config)


if __name__ == "__main__":
    main()
