import json
import re
from typing import Dict, List, Any, Optional

def parse_receipt_lines(formatted_lines: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parses spatial text lines into structured metadata and itemized product records.
    Returns both raw extracted results and transformed/arithmetically-inferred results.

    Args:
        formatted_lines (List[Dict[str, Any]]): Array of horizontally reconstructed line objects.
        config (Dict[str, Any]): Parsing parameters loaded from YAML configuration.

    Returns:
        Dict[str, Any]: Object containing both 'raw_parsed' and 'transformed' payloads.
    """
    raw_metadata: Dict[str, Any] = {}
    raw_items: List[Dict[str, Any]] = []
    unmapped_lines: List[str] = []
    
    fields_config = config.get("fields", {})
    items_section_config = config.get("items_section", {})
    item_patterns = config.get("item_patterns", [])
    
    start_trigger = items_section_config.get("start_trigger")
    end_trigger = items_section_config.get("end_trigger")
    
    currency = config.get("currency")
    
    in_items_section = False if start_trigger else True
    raw_lines = [line["line_text"] for line in formatted_lines]
    
    # -------------------------------------------------------------------------
    # STAGE 1: Pure Key-Value Field Metadata Extraction (Raw)
    # -------------------------------------------------------------------------
    for field_name, rules in fields_config.items():
        regex_pattern = rules.get("regex")
        group_id = rules.get("group", 1)
        split_trigger = rules.get("split_trigger")
        split_regex = rules.get("split_regex")
        
        for idx, line_text in enumerate(raw_lines):
            # 1. Match on same line
            if regex_pattern:
                match = re.search(regex_pattern, line_text)
                if match:
                    try:
                        val = match.group(group_id)
                        if val:
                            if currency and currency in val:
                                val = val.replace(currency, "").strip()
                            
                            val = val.strip()
                            
                            # Clean up OCR artifacts in specific fields
                            if field_name in ["unit_price", "total_price"]:
                                val = re.sub(r',\s+', ',', val)
                            elif field_name == "description":
                                val = val.replace("_", " ")
                                
                            raw_metadata[field_name] = val
                            break
                    except IndexError:
                        pass
            
            # 2. Match split across next or previous line
            if split_trigger and split_regex:
                if split_trigger in line_text:
                    val_match = None
                    split_dir = rules.get("split_direction", "down")
                    
                    if split_dir == "down" and idx + 1 < len(raw_lines):
                        next_line = raw_lines[idx + 1]
                        val_match = re.search(split_regex, next_line)
                    elif split_dir == "up" and idx - 1 >= 0:
                        prev_line = raw_lines[idx - 1]
                        val_match = re.search(split_regex, prev_line)
                    elif split_dir == "both":
                        if idx + 1 < len(raw_lines):
                            val_match = re.search(split_regex, raw_lines[idx + 1])
                        if not val_match and idx - 1 >= 0:
                            val_match = re.search(split_regex, raw_lines[idx - 1])
                        
                    if val_match:
                        try:
                            val = val_match.group(group_id)
                            if val:
                                if currency:
                                    val = val.replace(currency, '')
                                raw_metadata[field_name] = val.strip()
                                break
                        except IndexError:
                            pass

    # -------------------------------------------------------------------------
    # STAGE 2: Itemized Line Extraction & Multi-line Merging (Raw)
    # -------------------------------------------------------------------------
    pending_description = None

    for line_info in formatted_lines:
        line_text = line_info["line_text"].strip()
        
        # Check start section trigger
        if start_trigger and not in_items_section:
            if re.search(start_trigger, line_text):
                in_items_section = True
                unmapped_lines.append(line_text)
                continue
                
        # Check end section trigger
        if end_trigger and in_items_section:
            if re.search(end_trigger, line_text):
                in_items_section = False
                if pending_description:
                    raw_items.append({
                        "quantity": None,
                        "unit_price": None,
                        "total_price": None,
                        "description": pending_description
                    })
                    pending_description = None
                
        if in_items_section:
            matched_item = False
            
            for pattern_entry in item_patterns:
                pattern = pattern_entry.get("regex")
                if not pattern:
                    continue
                
                match = re.match(pattern, line_text)
                if match:
                    item_dict = match.groupdict()
                    
                    # Case A: Line contains price/quantity breakdown
                    if item_dict.get("unit_price") is not None:
                        if currency:
                            item_dict["unit_price"] = item_dict["unit_price"].replace(currency, '').strip()
                        if item_dict.get("total_price") and currency:
                            item_dict["total_price"] = item_dict["total_price"].replace(currency, '').strip()
                        
                        # Clean OCR spaces in prices
                        item_dict["unit_price"] = re.sub(r',\s+', ',', item_dict["unit_price"])
                        if item_dict.get("total_price"):
                            item_dict["total_price"] = re.sub(r',\s+', ',', item_dict["total_price"])

                            
                        if pending_description:
                            item_dict["description"] = pending_description
                            pending_description = None
                        
                        # Infer total price if missing from the OCR
                        if not item_dict.get("total_price") and item_dict.get("quantity") and item_dict.get("unit_price"):
                            try:
                                qty = int(item_dict["quantity"])
                                u_price = float(item_dict["unit_price"].replace(',', '.'))
                                item_dict["total_price"] = f"{(qty * u_price):.2f}".replace('.', ',')
                            except ValueError:
                                pass
                                
                        raw_items.append(item_dict)
                        matched_item = True
                        break
                    
                    # Case B: Line contains description only
                    elif item_dict.get("description") is not None and item_dict.get("unit_price") is None:
                        # Clean OCR underscores in descriptions
                        clean_desc = item_dict["description"].replace("_", " ")
                        if pending_description:
                            pending_description += f" {clean_desc}"
                        else:
                            pending_description = clean_desc
                        matched_item = True
                        break
            
            if not matched_item:
                unmapped_lines.append(line_text)
        else:
            unmapped_lines.append(line_text)

    if pending_description:
        raw_items.append({
            "quantity": None,
            "unit_price": None,
            "total_price": None,
            "description": pending_description
        })
        pending_description = None

    raw_output_data = {
        "items": json.loads(json.dumps(raw_items)),
        "subtotal": raw_metadata.get("subtotal"),
        "tax": raw_metadata.get("tax"),
        "total": raw_metadata.get("total")
    }
    if currency:
        raw_output_data["currency"] = currency

    # -------------------------------------------------------------------------
    # STAGE 3: Multi-Directional Arithmetic Integrity Engine (Transformed)
    # -------------------------------------------------------------------------
    transformed_metadata = dict(raw_metadata)
    transformed_items = json.loads(json.dumps(raw_items))

    def parse_float(val: Optional[str]) -> Optional[float]:
        if val is None:
            return None
        try:
            return float(str(val).replace(',', '.').replace('$', '').strip())
        except ValueError:
            return None

    subtotal = parse_float(transformed_metadata.get("subtotal"))
    tax = parse_float(transformed_metadata.get("tax"))
    total = parse_float(transformed_metadata.get("total"))

    # Extract item prices
    item_prices = [parse_float(i.get("total_price")) for i in transformed_items]
    has_unpriced_items = any(p is None for p in item_prices)
    valid_item_prices = [p for p in item_prices if p is not None]

    # Rule 1: Deduce missing single item price if Subtotal is known
    missing_items = [i for i in transformed_items if parse_float(i.get("total_price")) is None]
    if len(missing_items) == 1 and subtotal is not None and subtotal > sum(valid_item_prices):
        inferred_item_price = round(subtotal - sum(valid_item_prices), 2)
        missing_items[0]["total_price"] = f"{inferred_item_price:.2f}"
        missing_items[0]["inferred"] = True
        item_prices = [parse_float(i.get("total_price")) for i in transformed_items]
        has_unpriced_items = any(p is None for p in item_prices)
        valid_item_prices = [p for p in item_prices if p is not None]

    # Rule 2: Calculate missing Subtotal from sum of items (if ALL items have valid prices)
    if subtotal is None and not has_unpriced_items and len(valid_item_prices) > 0:
        subtotal = round(sum(valid_item_prices), 2)
        transformed_metadata["subtotal"] = f"{subtotal:.2f}"
        transformed_metadata["subtotal_inferred"] = True

    # Rule 3: Infer missing Subtotal (Subtotal = Total - Tax)
    if subtotal is None and total is not None and tax is not None:
        subtotal = round(total - tax, 2)
        transformed_metadata["subtotal"] = f"{subtotal:.2f}"
        transformed_metadata["subtotal_inferred"] = True

    # Rule 4: Infer missing Total (Total = Subtotal + Tax)
    if total is None and subtotal is not None and tax is not None:
        total = round(subtotal + tax, 2)
        transformed_metadata["total"] = f"{total:.2f}"
        transformed_metadata["total_inferred"] = True

    # Rule 5: Infer missing Tax (Tax = Total - Subtotal)
    if tax is None and total is not None and subtotal is not None:
        tax = round(total - subtotal, 2)
        transformed_metadata["tax"] = f"{tax:.2f}"
        transformed_metadata["tax_inferred"] = True

    transformed_output_data = {
        "items": transformed_items,
        "subtotal": transformed_metadata.get("subtotal"),
        "tax": transformed_metadata.get("tax"),
        "total": transformed_metadata.get("total")
    }
    if currency:
        transformed_output_data["currency"] = currency

    return {
        "raw_parsed": raw_output_data,
        "transformed": transformed_output_data
    }
