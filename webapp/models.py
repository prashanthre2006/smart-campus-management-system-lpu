from django.conf import settings
from django.db import models
from django.utils import timezone
import uuid


class UserProfile(models.Model):
    ROLE_CHOICES = [
        ("admin", "Admin"),
        ("faculty", "Faculty"),
        ("student", "Student"),
    ]
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default="student")

    def __str__(self):
        return f"{self.user.username} ({self.role})"


class Student(models.Model):
    name = models.CharField(max_length=120, unique=True)
    reg_no = models.CharField(max_length=40, blank=True)
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class ClassSection(models.Model):
    name = models.CharField(max_length=100)
    program = models.CharField(max_length=100, blank=True)
    year = models.CharField(max_length=20, blank=True)
    semester = models.CharField(max_length=20, blank=True)

    def __str__(self):
        return self.name


class Subject(models.Model):
    name = models.CharField(max_length=120)
    code = models.CharField(max_length=40, blank=True)

    def __str__(self):
        return self.name


class Enrollment(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    class_section = models.ForeignKey(ClassSection, on_delete=models.CASCADE)
    subject = models.ForeignKey(Subject, on_delete=models.SET_NULL, null=True, blank=True)

    class Meta:
        unique_together = ("student", "class_section", "subject")


class SchedulePeriod(models.Model):
    name = models.CharField(max_length=60)
    start_time = models.TimeField()
    end_time = models.TimeField()
    days_of_week = models.CharField(max_length=20, help_text="CSV like Mon,Tue,Wed")

    def __str__(self):
        return self.name


class Device(models.Model):
    name = models.CharField(max_length=100)
    location = models.CharField(max_length=120, blank=True)
    is_active = models.BooleanField(default=True)
    kiosk_id = models.CharField(max_length=80, unique=True, blank=True)
    last_seen_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=40, default="online")

    def __str__(self):
        return self.name


class Attendance(models.Model):
    STATUS_CHOICES = [
        ("present", "Present"),
        ("absent", "Absent"),
        ("late", "Late"),
    ]
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    date = models.DateField()
    period_name = models.CharField(max_length=60)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="present")
    device = models.ForeignKey(Device, on_delete=models.SET_NULL, null=True, blank=True)
    marked_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("student", "date", "period_name")


class AuditLog(models.Model):
    actor = models.CharField(max_length=120)
    action = models.CharField(max_length=80)
    detail = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)


class Faculty(models.Model):
    name = models.CharField(max_length=120)
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True)
    department = models.CharField(max_length=120, blank=True)

    def __str__(self):
        return self.name


class Course(models.Model):
    name = models.CharField(max_length=120)
    code = models.CharField(max_length=40, blank=True)

    def __str__(self):
        return self.name


class SubjectOffering(models.Model):
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    class_section = models.ForeignKey(ClassSection, on_delete=models.CASCADE)
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE)
    faculty = models.ForeignKey(Faculty, on_delete=models.SET_NULL, null=True, blank=True)

    class Meta:
        unique_together = ("course", "class_section", "subject")


class TimetableEntry(models.Model):
    DAYS = [
        ("Mon", "Mon"),
        ("Tue", "Tue"),
        ("Wed", "Wed"),
        ("Thu", "Thu"),
        ("Fri", "Fri"),
        ("Sat", "Sat"),
        ("Sun", "Sun"),
    ]
    day = models.CharField(max_length=3, choices=DAYS)
    period = models.ForeignKey(SchedulePeriod, on_delete=models.CASCADE)
    offering = models.ForeignKey(SubjectOffering, on_delete=models.CASCADE)
    room = models.CharField(max_length=60, blank=True)

    class Meta:
        unique_together = ("day", "period", "offering")


class CampusBlock(models.Model):
    name = models.CharField(max_length=120)
    code = models.CharField(max_length=40, unique=True)
    capacity = models.PositiveIntegerField(default=0)

    def __str__(self):
        return f"{self.code} - {self.name}"


class Classroom(models.Model):
    block = models.ForeignKey(CampusBlock, on_delete=models.CASCADE, related_name="classrooms")
    name = models.CharField(max_length=120)
    code = models.CharField(max_length=40, unique=True)
    capacity = models.PositiveIntegerField(default=0)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return f"{self.code} ({self.block.code})"


class BreakTimeSlot(models.Model):
    name = models.CharField(max_length=80)
    start_time = models.TimeField()
    end_time = models.TimeField()
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ["start_time"]

    def __str__(self):
        return self.name


class FoodItem(models.Model):
    CATEGORY_CHOICES = [
        ("Breakfast", "Breakfast"),
        ("Main Course", "Main Course"),
        ("Snacks", "Snacks"),
        ("Beverages", "Beverages"),
        ("Desserts", "Desserts"),
    ]
    name = models.CharField(max_length=120)
    category = models.CharField(max_length=40, choices=CATEGORY_CHOICES, default="Main Course")
    description = models.TextField(blank=True)
    price = models.DecimalField(max_digits=8, decimal_places=2)
    is_available = models.BooleanField(default=True)
    prep_time_mins = models.PositiveIntegerField(default=10)

    def __str__(self):
        return self.name


class FoodOrder(models.Model):
    ORDER_MODE_CHOICES = [
        ("dining", "Dining"),
        ("delivery", "Delivery"),
    ]
    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("confirmed", "Confirmed"),
        ("preparing", "Preparing"),
        ("ready", "Ready"),
        ("collected", "Collected"),
        ("cancelled", "Cancelled"),
    ]
    student = models.ForeignKey(Student, on_delete=models.SET_NULL, null=True, blank=True)
    slot = models.ForeignKey(BreakTimeSlot, on_delete=models.SET_NULL, null=True, blank=True)
    order_mode = models.CharField(max_length=20, choices=ORDER_MODE_CHOICES, default="dining")
    delivery_location = models.CharField(max_length=200, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    ordered_at = models.DateTimeField(default=timezone.now)
    total_amount = models.DecimalField(max_digits=10, decimal_places=2, default=0)

    class Meta:
        ordering = ["-ordered_at"]

    def __str__(self):
        return f"Order #{self.id}"


class FoodOrderItem(models.Model):
    order = models.ForeignKey(FoodOrder, on_delete=models.CASCADE, related_name="items")
    food_item = models.ForeignKey(FoodItem, on_delete=models.CASCADE)
    quantity = models.PositiveIntegerField(default=1)
    unit_price = models.DecimalField(max_digits=8, decimal_places=2)

    @property
    def line_total(self):
        return self.quantity * self.unit_price

    def __str__(self):
        return f"{self.food_item.name} x {self.quantity}"


class Payment(models.Model):
    METHOD_CHOICES = [
        ("qr", "QR"),
    ]
    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("paid", "Paid"),
        ("failed", "Failed"),
        ("refunded", "Refunded"),
    ]
    order = models.OneToOneField(FoodOrder, on_delete=models.CASCADE, related_name="payment")
    method = models.CharField(max_length=20, choices=METHOD_CHOICES, default="qr")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    transaction_id = models.CharField(max_length=80, blank=True)
    paid_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-id"]

    def __str__(self):
        return f"Payment #{self.id} ({self.status})"


class MakeupClassSession(models.Model):
    faculty = models.ForeignKey(Faculty, on_delete=models.SET_NULL, null=True, blank=True)
    course = models.ForeignKey(Course, on_delete=models.SET_NULL, null=True, blank=True)
    class_section = models.ForeignKey(ClassSection, on_delete=models.SET_NULL, null=True, blank=True)
    date = models.DateField(default=timezone.now)
    start_time = models.TimeField()
    end_time = models.TimeField()
    remedial_code = models.CharField(max_length=12, unique=True, blank=True)
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-date", "-start_time"]

    def save(self, *args, **kwargs):
        if not self.remedial_code:
            self.remedial_code = uuid.uuid4().hex[:8].upper()
        super().save(*args, **kwargs)

    def __str__(self):
        return f"Makeup {self.date} ({self.remedial_code})"


class MakeupAttendance(models.Model):
    MODE_CHOICES = [
        ("code", "Code"),
        ("face", "Face"),
    ]
    session = models.ForeignKey(MakeupClassSession, on_delete=models.CASCADE, related_name="attendance")
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    mode = models.CharField(max_length=10, choices=MODE_CHOICES, default="code")
    marked_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("session", "student")
        ordering = ["-marked_at"]

    def __str__(self):
        return f"{self.student.name} - {self.session.remedial_code}"
