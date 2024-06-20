import json
from pathlib import Path
from typing import Optional, List, Union, Dict, Any

from moonwatcher.utils.data import OPERATOR_DICT
from moonwatcher.dataset.dataset import MoonwatcherDataset, Slice
from moonwatcher.dataset.metadata import ATTRIBUTE_FUNCTIONS
from moonwatcher.model.model import MoonwatcherModel
from moonwatcher.metric import calculate_metric
from moonwatcher.utils.helpers import get_current_timestamp
from moonwatcher.utils.api_connector import upload_if_possible
from moonwatcher.utils.data import DataType
from moonwatcher.base.base import MoonwatcherObject
from moonwatcher.utils.data import Task


class Check(MoonwatcherObject):
    def __init__(
        self,
        name: str,
        dataset_or_slice: Union[MoonwatcherDataset, Slice],
        metric: str,
        metric_parameters: Optional[Dict] = None,
        metric_class: str = None,
        description: Optional[str] = None,
        metadata: Dict[str, Any] = None,
        operator: Optional[str] = None,
        value: Optional[float] = None,
    ):
        """
         Creates a check

        :param name: the name you want to give this check
        :param dataset_or_slice: the dataset or slice to use
        :param metric: the metric to apply (torchmetric)
        :param metric_parameters: optional parameters for the metrics
        :param description: optional description of the check
        :param metadata: optional tags for the check
        :param operator: optional if you want to test, compare symbol like >, >= etc.
        :param value: optional value to compare to
        """
        MoonwatcherObject.__init__(self, name=name, datatype=DataType.CHECK)
        self.testing = False
        self.description = description
        self.metadata = metadata
        self.operator_str = operator
        if operator is not None and value is not None:
            self.testing = True
        self.operator = operator
        self.value = value
        self.dataset_or_slice = dataset_or_slice
        self.metric = metric
        self.metric_parameters = metric_parameters
        self.metric_class = metric_class
        self.store()

    def _upload(self):
        if self.metric_parameters is not None:
            metric_dict = {k: self.metric_parameters[k] for k in self.metric_parameters}
        else:
            metric_dict = {}
        metric_dict["name"] = self.metric

        data = {
            "name": self.name,
            "description": self.description,
            "timestamp": get_current_timestamp(),
            "metadata": self.metadata,
            "testing": self.testing,
            "operator": self.operator_str,
            "value": self.value,
            "dataset_name": (
                self.dataset_or_slice.dataset_name
                if isinstance(self.dataset_or_slice, Slice)
                else self.dataset_or_slice.name
            ),
            "slice_name": (
                self.dataset_or_slice.name
                if isinstance(self.dataset_or_slice, Slice)
                else None
            ),
            "metric": metric_dict,
        }
        return upload_if_possible(datatype=DataType.CHECK.value, data=data)

    def __call__(
        self, model: MoonwatcherModel, show=False, save_report=False
    ) -> Dict[str, Any]:
        result = calculate_metric(
            model=model,
            dataset_or_slice=self.dataset_or_slice,
            metric=self.metric,
            metric_parameters=self.metric_parameters,
            metric_class=self.metric_class,
        )
        report = {
            "check_name": self.name,
            "dataset_name": (
                self.dataset_or_slice.dataset_name
                if isinstance(self.dataset_or_slice, Slice)
                else self.dataset_or_slice.name
            ),
            "slice_name": (
                self.dataset_or_slice.name
                if isinstance(self.dataset_or_slice, Slice)
                else None
            ),
            "model_name": model.name,
            "metric": self.metric,
            "operator": self.operator_str,
            "value": self.value,
            "result": result,
            "success": (
                OPERATOR_DICT[self.operator](result, self.value)
                if self.testing
                else None
            ),
            "timestamp": get_current_timestamp(),
        }

        if show:
            visualize_report(report)
        self.upload_if_not()
        self._upload_report(report=report)

        if save_report:
            with open(
                f"check_{self.name}_report_for_{model.name}.json", "w", encoding="utf-8"
            ) as f:
                json.dump(obj=report, fp=f, indent=4)
        return report

    def _upload_report(self, report):
        data = {
            "model_name": report["model_name"],
            "check_name": self.name,
            "timestamp": get_current_timestamp(),
            "result": report["result"],
            "passed": report["success"],
        }
        return upload_if_possible(datatype=DataType.CHECK_REPORT.value, data=data)


class CheckSuite(MoonwatcherObject):
    def __init__(
        self,
        name: str,
        checks: List[Check],
        description: Optional[str] = None,
        metadata: Dict[str, Any] = None,
        show=False,
    ):
        """
        Creates a checksuite

        :param name: name you want to give the checksuite
        :param checks: list of checks that are included
        :param description: optional description
        :param metadata: optional tags
        :param show: whether to print results in commandline
        """
        MoonwatcherObject.__init__(self, name=name, datatype=DataType.CHECK)
        self.description = description
        self.metadata = metadata
        self.checks = checks
        self.testing = False
        for check in self.checks:
            if check.testing:
                self.testing = True
                break
        self.show = show
        self.store()

    def _upload(self):
        data = {
            "name": self.name,
            "description": self.description,
            "timestamp": get_current_timestamp(),
            "metadata": self.metadata,
            "testing": self.testing,
            "check_names": [check.name for check in self.checks],
        }
        return upload_if_possible(datatype=DataType.CHECKSUITE.value, data=data)

    def __call__(self, model: MoonwatcherModel, show=None, save_report=False):
        total_success = None
        if self.testing:
            total_success = True

        reports = []

        show = self.show if show is None else show
        for check in self.checks:
            report = check(model)
            reports.append(report)
            if check.testing:
                if not report["success"]:
                    total_success = False

        checksuite_report = {
            "checksuite_name": self.name,
            "model_name": model.name,
            "timestamp": get_current_timestamp(),
            "success": total_success,
            "checks": reports,
        }

        if show:
            visualize_report(checksuite_report)
        self.upload_if_not()
        self._upload_report(report=checksuite_report)
        if save_report:
            with open(
                f"checksuite_{self.name}_report_for_{model.name}.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(obj=checksuite_report, fp=f, indent=4)

        return checksuite_report

    def _upload_report(self, report):
        data = {
            "model_name": report["model_name"],
            "checksuite_name": self.name,
            "timestamp": get_current_timestamp(),
            "passed": report["success"],
        }
        return upload_if_possible(datatype=DataType.CHECKSUITE_REPORT.value, data=data)


class ReportVisualizer:
    def __init__(self):
        self.pass_emoji = "\u2705"
        self.fail_emoji = "\u274C"
        self.BOLD = "\033[1m"
        self.END = "\033[0m"
        self.UNDERLINE = "\033[4m"
        self.RED = "\033[91m"
        self.GREEN = "\033[92m"

    def visualize(self, report, spacing=""):
        if report["success"] is not None:
            test_output_str = (
                f"{self.pass_emoji if report['success'] else self.fail_emoji} "
            )
        else:
            test_output_str = f"   "
        if "check_name" in report:
            name = report["check_name"]
        else:
            name = report["checksuite_name"]
        test_output_str = f"{spacing}{test_output_str}{name}"

        if "checks" in report:  # This is CheckSuite
            print(test_output_str)
            new_spacing = spacing + "    "
            for sub_report in report["checks"]:
                self.visualize(report=sub_report, spacing=new_spacing)
        else:  # This is Check
            report_result = f"{report['result']}".rjust(10)
            if report["success"] is not None:
                value = f"{report['value']}".ljust(10)
                operator = report["operator"]
                if operator == ">=":
                    operator = "≥"
                elif operator == "<=":
                    operator = "≤"
                elif operator == "==":
                    operator = "="
                elif operator == "!=":
                    operator = "≠"

                operator = f"{operator}".center(3)
                comparison = f"{operator} {value}"
                result = f"{self.GREEN if report['success'] else self.RED}{report_result}{self.END}"
            else:
                comparison = f""
                result = f" {report_result}"

            printed_set = report["dataset_name"]
            if report["slice_name"] is not None:
                printed_set = report["slice_name"]

            appendix = f"   ({report['metric']} on {printed_set})"
            print_statement = f"{test_output_str.ljust(40)} {self.BOLD}{result}{self.END} {comparison}{appendix}"
            print(print_statement)


def visualize_report(report):
    visualizer = ReportVisualizer()
    visualizer.visualize(report)


def automated_checking(
    mw_dataset: MoonwatcherDataset,
    mw_model: MoonwatcherModel,
    metadata_keys: List[str] = None,
    metadata_list: List[Dict[str, Any]] = None,
    slicing_conditions: List[Dict[str, Any]] = None,
    checks: List[Dict[str, Any]] = None,
    demo: bool = True,
):
    # Load demo configurations if specified
    if demo:
        if mw_dataset.task == Task.CLASSIFICATION.value:
            filename = "demo_classification.json"
        elif mw_dataset.task == Task.DETECTION.value:
            filename = "demo_detection.json"
        else:
            raise ValueError(f"Unsupported task: {mw_dataset.task}")

        cur_filepath = Path(__file__)
        with open(
            cur_filepath.parent / "configs" / filename, "r", encoding="utf-8"
        ) as file:
            config = json.load(file)

        metadata_keys = config["metadata_keys"]
        slicing_conditions = config["slicing_conditions"]
        checks = config["checks"]

    # Add Metadata
    for metadata_key in metadata_keys:
        if metadata_key in ATTRIBUTE_FUNCTIONS:
            mw_dataset.add_predefined_metadata(metadata_key)
        else:
            mw_dataset.add_metadata_from_groundtruths(metadata_key)
    if metadata_list:
        mw_dataset.add_metadata_from_list(metadata_list)

    # Create Slices
    mw_slices = []
    for slicing_condition in slicing_conditions:
        if slicing_condition["type"] == "threshold":
            mw_slice = mw_dataset.slice_by_threshold(
                slicing_condition["key"],
                slicing_condition["operator"],
                slicing_condition["value"],
            )
            mw_slices.append(mw_slice)
        elif slicing_condition["type"] == "percentile":
            mw_slice = mw_dataset.slice_by_percentile(
                slicing_condition["key"],
                slicing_condition["operator"],
                slicing_condition["value"],
            )
            mw_slices.append(mw_slice)
        else:
            slices = mw_dataset.slice_by_class(slicing_condition["key"])
            for mw_slice in slices:
                mw_slices.append(mw_slice)

    all_test_results = {}

    # Create and run checks for each slice
    for mw_slice in mw_slices:
        check_objects = []
        for check in checks:
            check_name = f"{check['name']}_{mw_slice.name}"
            new_check = Check(
                name=check_name,
                dataset_or_slice=mw_slice,
                metric=check["metric"],
                metric_class=check.get("metric_class", None),
                operator=check.get("operator", None),
                value=check.get("value", None),
            )
            check_objects.append(new_check)

        # Run Checks
        check_suite = CheckSuite(name=f"Test_{mw_slice.name}", checks=check_objects)
        test_results = check_suite(model=mw_model, show=True)
        all_test_results[mw_slice.name] = test_results

    # Create and run checks for the entire dataset
    check_objects = []
    for check in checks:
        check_name = f"{check['name']}_{mw_dataset.name}"
        new_check = Check(
            name=check_name,
            dataset_or_slice=mw_dataset,
            metric=check["metric"],
            metric_class=check.get("metric_class", None),
            operator=check.get("operator", None),
            value=check.get("value", None),
        )
        check_objects.append(new_check)

    entire_dataset_check_suite = CheckSuite(
        name=f"Test_{mw_dataset.name}", checks=check_objects
    )
    entire_dataset_test_results = entire_dataset_check_suite(model=mw_model, show=True)
    all_test_results[mw_dataset.name] = entire_dataset_test_results

    return all_test_results
