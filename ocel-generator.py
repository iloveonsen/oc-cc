from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import itertools
import random
import json


def iso(dt: datetime) -> str:
    """Formats a datetime object into an ISO 8601 string with Zulu timezone."""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

def uniq(series: itertools.count, prefix: str) -> str:
    """Generates a unique ID string from a counter and a prefix."""
    return f"{prefix}{next(series)}"


DEFAULT_EVENT_TYPES = [
    {"name": "OutpatientRegistration", "attributes": [{"name": "ReceptionDesk", "type": "string"}]},
    {"name": "InitialVisit",          "attributes": [{"name": "Department", "type": "string"}, {"name": "AttendingPhysician", "type": "string"}]},
    {"name": "ImagingTest",           "attributes": [{"name": "TestType", "type": "string"}]},
    {"name": "LabTest",               "attributes": [{"name": "TestType", "type": "string"}]},
    {"name": "FollowUpVisit",        "attributes": [{"name": "Assessment", "type": "string"}]},
    {"name": "JointConsult",          "attributes": [{"name": "CareTeam", "type": "string"}]},
    {"name": "MedicationTreatment",   "attributes": [{"name": "CycleNumber", "type": "integer"}]},
    {"name": "Payment",                "attributes": [{"name": "PaymentMethod", "type": "string"}, {"name": "PaymentAmount", "type": "integer"}]},
    {"name": "Discharge",              "attributes": []},
]

DEFAULT_OBJECT_TYPES = [
    {
        "name": "Patient",
        "attributes": [
            {"name": "patient_id", "type": "id", "format": "P-{:04d}"},
            {"name": "sex", "type": "string", "enum": ["M", "F"]},
            {"name": "age", "type": "integer", "semantics": "continuous", "range": {"min": 0, "max": 120}},
            {"name": "allergies", "type": "string", "example": "penicillin"},
            {"name": "contradindications", "type": "string", "example": "contrast"},
            {"name": "status", "type": "string", "enum": ["Outpatient", "Inpatient", "Discharged"]},
            {"name": "condition", "type": "string", "enum": ["Severe", "Critical", "Stable"]},
        ],
    },
    {
        "name": "Test",
        "attributes": [
            {"name": "test_id", "type": "id", "format": "T-{:04d}-{}"},
            {"name": "patient_id", "type": "id", "format": "P-{:04d}"},
            {"name": "modality", "type": "string", "enum": ["CT", "MRI", "PET", "Blood", "Pathology"]},
            {"name": "perform_time", "type": "time", "format": "%Y-%m-%dT%H:%M:%SZ", "timezone": "UTC"},
            {"name": "result_state", "type": "string", "enum": ["Unreviewed", "Reviewed"]},
            {"name": "findings", "type": "string", "example": "Suspicious for Cancer"},
        ],
    },
    {
        "name": "Encounter",
        "attributes": [
            {"name": "enc_id", "type": "id", "format": "E-{:04d}"},
            {"name": "patient_id", "type": "id", "format": "P-{:04d}"},
            {"name": "type", "type": "string", "enum": ["Outpatient", "Inpatient", "Emergency"]},
            {"name": "dept", "type": "string", "example": "Internal Medicine"},
            {"name": "start_time", "type": "time", "format": "%Y-%m-%dT%H:%M:%SZ", "timezone": "UTC"},
            {"name": "end_time", "type": "time", "format": "%Y-%m-%dT%H:%M:%SZ", "timezone": "UTC", "optional": True},
            {"name": "status", "type": "string", "enum": ["open", "closed"]},
            {"name": "reason", "type": "string", "optional": True, "example": "AbdominalPain"},
            {"name": "attending", "type": "string", "enum": ["Dr. Kim", "Dr. Cha", "Dr.Jung"]},
            {"name": "billing_status", "type": "string", "enum": ["unpaid", "partial", "paid"]},
            {"name": "billing_amount", "type": "integer", "semantics": "continuous", "unit": "KRW", "range": {"min": 0, "max": 10000000}},
            {"name": "payment_method", "type": "string", "enum": ["card", "cash", "transfer"], "optional": True},
            {"name": "billing_time", "type": "time", "format": "%Y-%m-%dT%H:%M:%SZ", "timezone": "UTC", "optional": True},
        ],
    },
    {
        "name": "Diagnosis",
        "attributes": [
            {"name": "dx_id", "type": "id", "format": "DX-{:04d}"},
            {"name": "patient_id", "type": "id", "format": "P-{:04d}"},
            {"name": "primary_site", "type": "string", "example": "Lung"},
            {"name": "stage", "type": "string", "enum": ["I", "II", "III", "IV"]},
            {"name": "confirm_time", "type": "time", "format": "%Y-%m-%dT%H:%M:%SZ", "timezone": "UTC"},
        ],
    },
    {
        "name": "TreatmentPlan",
        "attributes": [
            {"name": "plan_id", "type": "id", "format": "PLAN-{:04d}"},
            {"name": "patient_id", "type": "id", "format": "P-{:04d}"},
            {"name": "dx_id", "type": "id", "format": "DX-{:04d}"},
            {"name": "intent", "type": "string", "enum": ["curative", "palliative"]},
            {"name": "primary_modality", "type": "string", "example": "medication"},
            {"name": "regimen", "type": "string", "example": "FOLFOX"},
            {"name": "status", "type": "string", "enum": ["Planned", "Proceed", "Complete"]},
        ],
    },
    {
        "name": "Dose",
        "attributes": [
            {"name": "admin_id", "type": "id", "format": "D-{:04d}-{}"},
            {"name": "patient_id", "type": "id", "format": "P-{:04d}"},
            {"name": "plan_id", "type": "id", "format": "PLAN-{:04d}"},
            {"name": "drug_name", "type": "string", "example": "pembrolizumab"},
            {"name": "amount", "type": "integer", "semantics": "continuous", "unit": "mg", "range": {"min": 0, "max": 2000}},
            {"name": "route", "type": "string", "enum": ["IV", "PO"]},
            {"name": "given_time", "type": "time", "format": "%Y-%m-%dT%H:%M:%SZ", "timezone": "UTC"},
            {"name": "adverse_event", "type": "string", "example": "None"},
        ],
    },
]

DEFAULT_O2O_TYPES = [
    {"name": "Encounter-Patient", "sourceType": "Encounter", "targetType": "Patient", "attributes": []},
    {"name": "TreatmentPlan-Dose", "sourceType": "TreatmentPlan", "targetType": "Dose", "attributes": []},  # Note: sourceType changed to match object name
]


@dataclass
class OCELBuilder:
    event_types: List[Dict[str, Any]] = field(default_factory=lambda: DEFAULT_EVENT_TYPES)
    object_types: List[Dict[str, Any]] = field(default_factory=lambda: DEFAULT_OBJECT_TYPES)
    o2o_types:   List[Dict[str, Any]] = field(default_factory=lambda: DEFAULT_O2O_TYPES)

    _event_seq: itertools.count = field(default_factory=lambda: itertools.count(1))
    _obj_seq_by_type: Dict[str, itertools.count] = field(default_factory=dict)
    _o2o_seq: itertools.count = field(default_factory=lambda: itertools.count(1))
    obj_id_pad: int = 3

    _type_prefix: Dict[str, str] = field(default_factory=lambda: {
        "Patient": "pat",
        "Test": "test",
        "Encounter": "enc",
        "Diagnosis": "dx",
        "TreatmentPlan": "plan",
        "Dose": "med",
    })

    events: List[Dict[str, Any]] = field(default_factory=list)
    objects: List[Dict[str, Any]] = field(default_factory=list)
    object_relationship_types: List[Dict[str, Any]] = field(default_factory=list)
    object_relationships: List[Dict[str, Any]] = field(default_factory=list)

    base_time: datetime = field(default_factory=lambda: datetime(2025, 8, 1, 9, 0, 0))
    event_step_mean_minutes: float = 30.0
    event_step_std_minutes: float = 10.0
    rng_seed: int = 42

    def __post_init__(self):
        random.seed(self.rng_seed)
        np.random.seed(self.rng_seed)
        self.object_relationship_types = self.o2o_types.copy()

    def add_traces(self, traces: List[List[str]]) -> None:
        for idx, trace in enumerate(traces, start=1):
            self._add_single_trace(trace_index=idx, activities=trace)

    def to_json(self) -> Dict[str, Any]:
        return {
            "eventTypes": self.event_types,
            "objectTypes": self.object_types,
            "events": self.events,
            "objects": self.objects,
            "objectRelationshipTypes": self.object_relationship_types,
            "objectRelationships": self.object_relationships
        }

    def _add_single_trace(self, trace_index: int, activities: List[str]) -> None:
        t = self.base_time + timedelta(hours=(trace_index-1))
        minutes_to_add = np.random.normal(self.event_step_mean_minutes, self.event_step_std_minutes)
        step = timedelta(minutes=max(10, minutes_to_add))
        ctx = self._init_trace_objects(trace_index=trace_index, start_time=t)

        for i, act in enumerate(activities, start=1):
            # print(f"  Event {i}: {act} at {iso(t)}")
            if act == "OutpatientRegistration":
                self._emit_event_registration(ctx, t)
            elif act == "InitialVisit":
                self._emit_event_initial_consult(ctx, t)
            elif act == "ImagingTest":
                self._emit_event_imaging(ctx, t)
            elif act == "LabTest":
                self._emit_event_lab(ctx, t)
            elif act == "FollowUpVisit":
                self._emit_event_followup(ctx, t)
            elif act == "JointConsult":
                self._emit_event_mdt(ctx, t)
            elif act == "MedicationTreatment":
                self._emit_event_medication(ctx, t)
            elif act == "Payment":
                self._emit_event_payment(ctx, t)
            elif act == "Discharge":
                self._emit_event_discharge(ctx, t)
            else:
                raise ValueError(f"Unknown activity label: {act}")
            t += step
        
        self._emit_o2o(ctx)

    def _init_trace_objects(self, trace_index: int, start_time: datetime) -> Dict[str, Any]:
        ctx = {
            "trace_index": trace_index,
            "start_time": start_time,
            "patient_id": self._create_patient(trace_index, start_time),
            "visit_id": self._create_visit(trace_index, start_time),
            "diagnosis_id": None,
            "plan_id": None,
            "medication_ids": [],
            "test_ids": [], # Can have multiple tests
            "med_cycles": 0,
        }
        return ctx

    def _emit_event_registration(self, ctx, t):
        e = {
            "id": self._new_event_id(),
            "type": "OutpatientRegistration",
            "time": iso(t),
            "attributes": [{"name":"ReceptionDesk","value": f"Outpatient {(ctx['trace_index']%3)+1}"}],
            "relationships": [
                {"objectId": ctx["patient_id"], "qualifier": "subject"},
                {"objectId": ctx["visit_id"], "qualifier": "encounter"},
            ]
        }
        self.events.append(e)

    def _emit_event_initial_consult(self, ctx, t):
        doctor = random.choice(["Dr. Kim","Dr. Cha","Dr.Jung"])
        self._append_attr(ctx["visit_id"], "attending", iso(t), doctor)
        e = {
            "id": self._new_event_id(),
            "type": "InitialVisit",
            "time": iso(t),
            "attributes": [
                {"name":"Department","value": random.choice(["Oncology","Surgery"])},
                {"name":"AttendingPhysician","value": doctor}
            ],
            "relationships": [
                {"objectId": ctx["patient_id"], "qualifier": "subject"},
                {"objectId": ctx["visit_id"], "qualifier": "encounter"},
            ]
        }
        self.events.append(e)

    def _emit_event_imaging(self, ctx, t):
        modality = random.choice(["CT", "MRI", "PET"])
        test_id = self._create_test(ctx, t, modality)
        ctx["test_ids"].append(test_id)
        e = {
            "id": self._new_event_id(),
            "type": "ImagingTest",
            "time": iso(t),
            "attributes": [{"name":"TestType", "value": modality}],
            "relationships": [
                {"objectId": ctx["patient_id"], "qualifier": "subject"},
                {"objectId": ctx["visit_id"], "qualifier": "encounter"},
                {"objectId": test_id, "qualifier": "order"},
            ]
        }
        self.events.append(e)

    def _emit_event_lab(self, ctx, t):
        modality = random.choice(["Blood", "Pathology"])
        test_id = self._create_test(ctx, t, modality)
        ctx["test_ids"].append(test_id)
        e = {
            "id": self._new_event_id(),
            "type": "LabTest",
            "time": iso(t),
            "attributes": [{"name":"TestType", "value": modality}],
            "relationships": [
                {"objectId": ctx["patient_id"], "qualifier": "subject"},
                {"objectId": ctx["visit_id"], "qualifier": "encounter"},
                {"objectId": test_id, "qualifier": "order"},
            ]
        }
        self.events.append(e)

    def _emit_event_followup(self, ctx, t):
        # Mark all previous test results as "확인"
        for test_id in ctx["test_ids"]:
            self._append_attr(test_id, "result_state", iso(t), "Reviewed")
        
        e = {
            "id": self._new_event_id(),
            "type": "FollowUpVisit",
            "time": iso(t),
            "attributes": [{"name":"Assessment","value": random.choice(["AdditionalTesting","TreatmentPlanning","Monitoring"])}],
            "relationships": [
                {"objectId": ctx["patient_id"], "qualifier": "subject"},
                {"objectId": ctx["visit_id"], "qualifier": "encounter"},
            ]
        }
        # Link all tests that were reviewed in this event
        for test_id in ctx["test_ids"]:
            e["relationships"].append({"objectId": test_id, "qualifier": "reviewed_test"})
        self.events.append(e)

    def _emit_event_mdt(self, ctx, t):
        if ctx["diagnosis_id"] is None:
            ctx["diagnosis_id"] = self._create_diagnosis(ctx, t)
        if ctx["plan_id"] is None:
            ctx["plan_id"] = self._create_plan(ctx, t, ctx["diagnosis_id"])

        e = {
            "id": self._new_event_id(),
            "type": "MDT",
            "time": iso(t),
            "attributes": [{"name":"Care Team","value":"OncologySurgeryRadiology"}],
            "relationships": [
                {"objectId": ctx["patient_id"], "qualifier": "subject"},
                {"objectId": ctx["visit_id"], "qualifier": "encounter"},
                {"objectId": ctx["diagnosis_id"], "qualifier": "diagnosis"},
                {"objectId": ctx["plan_id"], "qualifier": "plan"},
            ]
        }
        self.events.append(e)

    def _emit_event_medication(self, ctx, t):
        if ctx["diagnosis_id"] is None:
            ctx["diagnosis_id"] = self._create_diagnosis(ctx, t)
        if ctx["plan_id"] is None:
            ctx["plan_id"] = self._create_plan(ctx, t, ctx["diagnosis_id"])
        
        med_id = self._create_medication(ctx, t, ctx["plan_id"])
        ctx["medication_ids"].append(med_id)
        ctx["med_cycles"] += 1
        self._append_attr(ctx["plan_id"], "status", iso(t), "Proceed")

        e = {
            "id": self._new_event_id(),
            "type": "MedicationTreatment",
            "time": iso(t),
            "attributes": [{"name":"CycleNumber","value": ctx["med_cycles"]}],
            "relationships": [
                {"objectId": ctx["patient_id"], "qualifier": "subject"},
                {"objectId": ctx["visit_id"], "qualifier": "encounter"},
                {"objectId": ctx["plan_id"], "qualifier": "plan"},
                {"objectId": med_id, "qualifier": "drug_admin"},
            ]
        }
        self.events.append(e)

    def _emit_event_payment(self, ctx, t):
        amount = random.randint(100000, 1000000)
        method = random.choice(["card", "cash", "transfer"])
        self._append_obj_attrs(ctx["visit_id"], [
            ("billing_amount", iso(t), amount),
            ("billing_status", iso(t), "paid"),
            ("payment_method", iso(t), method),
            ("billing_time", iso(t), iso(t))
        ])
        e = {
            "id": self._new_event_id(),
            "type": "Payment",
            "time": iso(t),
            "attributes": [{"name":"PaymentMethod", "value": method}, {"name":"PaymentAmount", "value": amount}],
            "relationships": [
                {"objectId": ctx["patient_id"], "qualifier": "subject"},
                {"objectId": ctx["visit_id"], "qualifier": "billing_encounter"},
            ]
        }
        self.events.append(e)

    def _emit_event_discharge(self, ctx, t):
        self._append_obj_attrs(ctx["visit_id"], [
            ("status", iso(t), "closed"),
            ("end_time", iso(t), iso(t))
        ])
        e = {
            "id": self._new_event_id(),
            "type": "Discharge",
            "time": iso(t),
            "attributes": [],
            "relationships": [
                {"objectId": ctx["patient_id"], "qualifier": "subject"},
                {"objectId": ctx["visit_id"], "qualifier": "encounter"},
            ]
        }
        self.events.append(e)
        
    # NEW CREATOR FUNCTIONS for the new object model
    def _create_patient(self, trace_index, t):
        oid = self._new_object_id("Patient")
        obj = {
            "id": oid, "type": "Patient",
            "attributes": [
                {"name": "patient_id", "time": iso(t), "value": f"P-{trace_index:04d}"},
                {"name": "sex", "time": iso(t), "value": random.choice(["M", "F"])},
                {"name": "age", "time": iso(t), "value": random.randint(20, 85)},
                {"name": "contradindications", "time": iso(t), "value": "none"},
                {"name": "allergies", "time": iso(t), "value": "none"},
                {"name": "status", "time": iso(t), "value": "Outpatient"},
                {"name": "condition", "time": iso(t), "value": "Stable"},
            ]
        }
        self.objects.append(obj)
        return oid
    
    def _create_visit(self, trace_index, t):
        oid = self._new_object_id("Encounter")
        obj = {
            "id": oid, "type": "Encounter",
            "attributes": [
                {"name": "enc_id", "time": iso(t), "value": f"E-{trace_index:04d}"},
                {"name": "patient_id", "time": iso(t), "value": f"P-{trace_index:04d}"},
                {"name": "type", "time": iso(t), "value": "Outpatient"},
                {"name": "dept", "time": iso(t), "value": "Oncology"},
                {"name": "start_time", "time": iso(t), "value": iso(t)},
                {"name": "status", "time": iso(t), "value": "open"},
                {"name": "reason", "time": iso(t), "value": random.choice(["StomachPain","ChestPain","ThroatPain"])},
                {"name": "billing_status", "time": iso(t), "value": "unpaid"},
            ]
        }
        self.objects.append(obj)
        return oid
        
    def _create_test(self, ctx, t, modality):
        # Custom ID generation for tests
        test_num_in_trace = len(ctx["test_ids"]) + 1
        oid = f"test-{ctx['trace_index']:03d}-{test_num_in_trace}"
        
        obj = {
            "id": oid, "type": "Test",
            "attributes": [
                {"name": "test_id", "time": iso(t), "value": f"T-{ctx['trace_index']:04d}-{modality}{test_num_in_trace}"},
                {"name": "patient_id", "time": iso(t), "value": f"P-{ctx['trace_index']:04d}"},
                {"name": "modality", "time": iso(t), "value": modality},
                {"name": "perform_time", "time": iso(t), "value": iso(t)},
                {"name": "result_state", "time": iso(t), "value": "Unreviewed"},
                {"name": "findings", "time": iso(t + timedelta(minutes=15)), "value": random.choice(["Suspicious", "Normal"])}
            ]
        }
        self.objects.append(obj)
        return oid
        
    def _create_diagnosis(self, ctx, t):
        oid = self._new_object_id("Diagnosis")
        obj = {
            "id": oid, "type": "Diagnosis",
            "attributes": [
                {"name": "dx_id", "time": iso(t), "value": f"DX-{ctx['trace_index']:04d}"},
                {"name": "patient_id", "time": iso(t), "value": f"P-{ctx['trace_index']:04d}"},
                {"name": "primary_site", "time": iso(t), "value": random.choice(["Lung", "Stomach", "Breast"])},
                {"name": "stage", "time": iso(t), "value": random.choice(["I", "II", "III", "IV"])},
                {"name": "confirm_time", "time": iso(t), "value": iso(t)},
            ]
        }
        self.objects.append(obj)
        return oid
        
    def _create_plan(self, ctx, t, diagnosis_id):
        oid = self._new_object_id("TreatmentPlan")
        dx_id_val = self._get_obj_attr(diagnosis_id, "dx_id")
        obj = {
            "id": oid, "type": "TreatmentPlan",
            "attributes": [
                {"name": "plan_id", "time": iso(t), "value": f"PLAN-{ctx['trace_index']:04d}"},
                {"name": "patient_id", "time": iso(t), "value": f"P-{ctx['trace_index']:04d}"},
                {"name": "dx_id", "time": iso(t), "value": dx_id_val},
                {"name": "intent", "time": iso(t), "value": random.choice(["curative", "palliative"])},
                {"name": "primary_modality", "time": iso(t), "value": "medication"},
                {"name": "regimen", "time": iso(t), "value": random.choice(["FOLFOX", "FOLFIRI", "CAPOX"])},
                {"name": "status", "time": iso(t), "value": "Planned"},
            ]
        }
        self.objects.append(obj)
        return oid

    def _create_medication(self, ctx, t, plan_id):
        # Custom ID generation for medications
        med_num_in_trace = len(ctx["medication_ids"]) + 1
        oid = f"med-{ctx['trace_index']:03d}-{med_num_in_trace}"
        plan_id_val = self._get_obj_attr(plan_id, "plan_id")
        
        obj = {
            "id": oid, "type": "Dose",
            "attributes": [
                {"name": "admin_id", "time": iso(t), "value": f"D-{ctx['trace_index']:04d}-{med_num_in_trace}"},
                {"name": "patient_id", "time": iso(t), "value": f"P-{ctx['trace_index']:04d}"},
                {"name": "plan_id", "time": iso(t), "value": plan_id_val},
                {"name": "drug_name", "time": iso(t), "value": random.choice(["pembrolizumab", "trastuzumab", "paclitaxel"])},
                {"name": "amount", "time": iso(t), "value": random.randint(100, 500)},
                {"name": "route", "time": iso(t), "value": "IV"},
                {"name": "given_time", "time": iso(t), "value": iso(t)},
                {"name": "adverse_event", "time": iso(t + timedelta(minutes=30)), "value": "None" if random.random() > 0.2 else "Nausea"},
            ]
        }
        self.objects.append(obj)
        return oid

    # REWRITTEN: O2O relationships are now much simpler.
    def _emit_o2o(self, ctx):
        def new_rel(rtype, time, src, tgt):
            rel = {
                "id": f"or{next(self._o2o_seq)}",
                "type": rtype,
                "time": iso(time),
                "sourceObject": src,
                "targetObject": tgt,
                "attributes": []
            }
            self.object_relationships.append(rel)

        # 방문-환자
        new_rel("Encounter-Patient", ctx["start_time"], ctx["visit_id"], ctx["patient_id"])

        # 치료계획-약물 (one plan can have multiple medication administrations)
        if ctx["plan_id"] and ctx["medication_ids"]:
            for med_id in ctx["medication_ids"]:
                 new_rel("TreatmentPlan-Dose", ctx["start_time"], ctx["plan_id"], med_id)

    # UTILITY FUNCTIONS (mostly unchanged)
    def _get_obj_attr(self, obj_id: str, attr_name: str) -> Any:
        for obj in reversed(self.objects):
            if obj["id"] == obj_id:
                vals = [a for a in obj["attributes"] if a["name"] == attr_name]
                return vals[-1]["value"] if vals else None
        return None

    def _append_attr(self, obj_id: str, name: str, time: str, value: Any):
        for obj in reversed(self.objects):
            if obj["id"] == obj_id:
                obj["attributes"].append({"name": name, "time": time, "value": value})
                return
        raise KeyError(f"Object {obj_id} not found")

    def _append_obj_attrs(self, obj_id: str, triples: List[Tuple[str, str, Any]]):
        for n, t, v in triples:
            self._append_attr(obj_id, n, t, v)

    def _next_obj_suffix(self, type_name: str) -> int:
        if type_name not in self._obj_seq_by_type:
            self._obj_seq_by_type[type_name] = itertools.count(1)
        return next(self._obj_seq_by_type[type_name])

    def _new_object_id(self, type_name: str) -> str:
        n = self._next_obj_suffix(type_name)
        pad = str(n).zfill(self.obj_id_pad)
        base = self._type_prefix.get(type_name, "obj")
        return f"{base}-{pad}"

    def _new_event_id(self) -> str:
        return uniq(self._event_seq, "e")


if __name__ == "__main__":
    # Example traces can remain the same, as they represent the sequence of activities.
    # traces = [
    #     ["외래 접수", "초진", "영상 검사", "재진", "협진", "약물 치료", "수납", "퇴원"],
    #     ["외래 접수", "초진", "영상 검사", "재진", "협진", "수납", "퇴원"],

    #     ["외래 접수", "초진", "실험실 검사", "재진", "협진", "약물 치료", "수납", "퇴원"],
    #     ["외래 접수", "초진", "실험실 검사", "재진", "협진", "수납", "퇴원"],

    #     ["외래 접수", "초진", "영상 검사", "실험실 검사", "재진", "협진", "약물 치료", "수납", "퇴원"],
    #     ["외래 접수", "초진", "영상 검사", "실험실 검사", "재진", "협진", "수납", "퇴원"],

    #     ["외래 접수", "초진", "실험실 검사", "영상 검사", "재진", "협진", "약물 치료", "수납", "퇴원"],
    #     ["외래 접수", "초진", "실험실 검사", "영상 검사", "재진", "협진", "수납", "퇴원"],

    #     ["외래 접수", "초진", "영상 검사", "재진", "영상 검사", "재진", "협진", "약물 치료", "수납", "퇴원"],
    #     ["외래 접수", "초진", "영상 검사", "재진", "영상 검사", "재진", "협진", "수납", "퇴원"],

    #     ["외래 접수", "초진", "영상 검사", "재진", "실험실 검사", "재진", "협진", "약물 치료", "수납", "퇴원"],
    #     ["외래 접수", "초진", "영상 검사", "재진", "실험실 검사", "재진", "협진", "수납", "퇴원"],

    #     ["외래 접수", "초진", "영상 검사", "재진", "영상 검사", "실험실 검사", "재진", "협진", "약물 치료", "수납", "퇴원"],
    #     ["외래 접수", "초진", "영상 검사", "재진", "영상 검사", "실험실 검사", "재진", "협진", "수납", "퇴원"],

    #     ["외래 접수", "초진", "영상 검사", "재진", "실험실 검사", "영상 검사", "재진", "협진", "약물 치료", "수납", "퇴원"],
    #     ["외래 접수", "초진", "영상 검사", "재진", "실험실 검사", "영상 검사", "재진", "협진", "수납", "퇴원"],

    #     ["외래 접수", "초진", "실험실 검사", "재진", "영상 검사", "재진", "협진", "약물 치료", "수납", "퇴원"],
    #     ["외래 접수", "초진", "실험실 검사", "재진", "영상 검사", "재진", "협진", "수납", "퇴원"],

    #     ["외래 접수", "초진", "실험실 검사", "재진", "실험실 검사", "재진", "협진", "약물 치료", "수납", "퇴원"],
    #     ["외래 접수", "초진", "실험실 검사", "재진", "실험실 검사", "재진", "협진", "수납", "퇴원"],

    #     ["외래 접수", "초진", "실험실 검사", "재진", "영상 검사", "실험실 검사", "재진", "협진", "약물 치료", "수납", "퇴원"],
    #     ["외래 접수", "초진", "실험실 검사", "재진", "영상 검사", "실험실 검사", "재진", "협진", "수납", "퇴원"],

    #     ["외래 접수", "초진", "실험실 검사", "재진", "실험실 검사", "영상 검사", "재진", "협진", "약물 치료", "수납", "퇴원"],
    #     ["외래 접수", "초진", "실험실 검사", "재진", "실험실 검사", "영상 검사", "재진", "협진", "수납", "퇴원"],

    #     ["외래 접수", "초진", "영상 검사", "실험실 검사", "재진", "영상 검사", "재진", "협진", "약물 치료", "수납", "퇴원"],
    #     ["외래 접수", "초진", "영상 검사", "실험실 검사", "재진", "영상 검사", "재진", "협진", "수납", "퇴원"],

    #     ["외래 접수", "초진", "영상 검사", "실험실 검사", "재진", "실험실 검사", "재진", "협진", "약물 치료", "수납", "퇴원"],
    #     ["외래 접수", "초진", "영상 검사", "실험실 검사", "재진", "실험실 검사", "재진", "협진", "수납", "퇴원"],

    #     ["외래 접수", "초진", "영상 검사", "실험실 검사", "재진", "영상 검사", "실험실 검사", "재진", "협진", "약물 치료", "수납", "퇴원"],
    #     ["외래 접수", "초진", "영상 검사", "실험실 검사", "재진", "영상 검사", "실험실 검사", "재진", "협진", "수납", "퇴원"],

    #     ["외래 접수", "초진", "영상 검사", "실험실 검사", "재진", "실험실 검사", "영상 검사", "재진", "협진", "약물 치료", "수납", "퇴원"],
    #     ["외래 접수", "초진", "영상 검사", "실험실 검사", "재진", "실험실 검사", "영상 검사", "재진", "협진", "수납", "퇴원"],

    #     ["외래 접수", "초진", "실험실 검사", "영상 검사", "재진", "영상 검사", "재진", "협진", "약물 치료", "수납", "퇴원"],
    #     ["외래 접수", "초진", "실험실 검사", "영상 검사", "재진", "영상 검사", "재진", "협진", "수납", "퇴원"],

    #     ["외래 접수", "초진", "실험실 검사", "영상 검사", "재진", "실험실 검사", "재진", "협진", "약물 치료", "수납", "퇴원"],
    #     ["외래 접수", "초진", "실험실 검사", "영상 검사", "재진", "실험실 검사", "재진", "협진", "수납", "퇴원"],

    #     ["외래 접수", "초진", "실험실 검사", "영상 검사", "재진", "영상 검사", "실험실 검사", "재진", "협진", "약물 치료", "수납", "퇴원"],
    #     ["외래 접수", "초진", "실험실 검사", "영상 검사", "재진", "영상 검사", "실험실 검사", "재진", "협진", "수납", "퇴원"],

    #     ["외래 접수", "초진", "실험실 검사", "영상 검사", "재진", "실험실 검사", "영상 검사", "재진", "협진", "약물 치료", "수납", "퇴원"],
    #     ["외래 접수", "초진", "실험실 검사", "영상 검사", "재진", "실험실 검사", "영상 검사", "재진", "협진", "수납", "퇴원"]
    # ]

    # no_seq_no_const
    # traces = [ 
    #     ["외래 접수", "초진", "영상 검사", "실험실 검사", "재진", "실험실 검사", "재진", "협진", "약물 치료", "수납", "퇴원"]
    # ]

    # no_seq_yes_const
    # traces = [
    #     ["외래 접수", "초진", "영상 검사", "실험실 검사", "재진", "실험실 검사", "재진", "협진", "약물 치료", "수납", "퇴원"]
    # ]

    # yes_seq_no_const
    # traces = [
    #     ["외래 접수", "초진", "영상 검사", "실험실 검사", "재진", "실험실 검사", "약물 치료", "협진", "수납", "퇴원"]
    # ]

    # yes_seq_yes_const
    traces = [
        ["외래 접수", "초진", "영상 검사", "실험실 검사", "재진", "실험실 검사", "재진", "협진", "퇴원"]
    ]

    event_map = {
        "외래 접수": "OutpatientRegistration",
        "초진": "InitialVisit",
        "영상 검사": "ImagingTest",
        "실험실 검사": "LabTest",
        "재진": "FollowUpVisit",
        "협진": "JointConsult",
        "약물 치료": "MedicationTreatment",
        "수납": "Payment",
        "퇴원": "Discharge",
    }

    unknown = {evt for trace in traces for evt in trace if evt not in event_map}
    if unknown:
        raise ValueError(f"Unmapped labels found: {sorted(unknown)}")
    
    translated_traces = [[event_map[evt] for evt in trace] for trace in traces]

    builder = OCELBuilder(
        rng_seed=123,
        event_step_mean_minutes=45.0,
        event_step_std_minutes=15.0
    )
    builder.add_traces(translated_traces)
    ocel = builder.to_json()

    with open("yes_seq_yes_const_eng.json", "w+", encoding="utf-8") as f:
        json.dump(ocel, f, ensure_ascii=False, indent=4)

    print("Generated OCEL with",
          len(ocel["events"]), "events,",
          len(ocel["objects"]), "objects,",
          len(ocel["objectRelationships"]), "O2O relations.")
