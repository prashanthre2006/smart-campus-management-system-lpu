from django.contrib import admin

from .models import (
    Attendance,
    AuditLog,
    BreakTimeSlot,
    CampusBlock,
    ClassSection,
    Classroom,
    Course,
    Device,
    Enrollment,
    Faculty,
    FoodItem,
    FoodOrder,
    FoodOrderItem,
    MakeupAttendance,
    MakeupClassSession,
    Payment,
    SchedulePeriod,
    Student,
    Subject,
    SubjectOffering,
    TimetableEntry,
    UserProfile
)


@admin.register(Student)
class StudentAdmin(admin.ModelAdmin):
    list_display = ("name", "reg_no", "created_at")
    search_fields = ("name", "reg_no")


@admin.register(ClassSection)
class ClassSectionAdmin(admin.ModelAdmin):
    list_display = ("name", "program", "year", "semester")


@admin.register(Subject)
class SubjectAdmin(admin.ModelAdmin):
    list_display = ("name", "code")


@admin.register(Enrollment)
class EnrollmentAdmin(admin.ModelAdmin):
    list_display = ("student", "class_section", "subject")


@admin.register(SchedulePeriod)
class SchedulePeriodAdmin(admin.ModelAdmin):
    list_display = ("name", "start_time", "end_time", "days_of_week")


@admin.register(Device)
class DeviceAdmin(admin.ModelAdmin):
    list_display = ("name", "location", "is_active", "kiosk_id", "status", "last_seen_at")


@admin.register(Attendance)
class AttendanceAdmin(admin.ModelAdmin):
    list_display = ("student", "date", "period_name", "status", "device", "marked_at")
    list_filter = ("date", "period_name", "status")


@admin.register(AuditLog)
class AuditLogAdmin(admin.ModelAdmin):
    list_display = ("actor", "action", "created_at")


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ("user", "role")


@admin.register(Faculty)
class FacultyAdmin(admin.ModelAdmin):
    list_display = ("name", "department")


@admin.register(Course)
class CourseAdmin(admin.ModelAdmin):
    list_display = ("name", "code")


@admin.register(SubjectOffering)
class SubjectOfferingAdmin(admin.ModelAdmin):
    list_display = ("course", "class_section", "subject", "faculty")


@admin.register(TimetableEntry)
class TimetableEntryAdmin(admin.ModelAdmin):
    list_display = ("day", "period", "offering", "room")


@admin.register(CampusBlock)
class CampusBlockAdmin(admin.ModelAdmin):
    list_display = ("code", "name", "capacity")
    search_fields = ("code", "name")


@admin.register(Classroom)
class ClassroomAdmin(admin.ModelAdmin):
    list_display = ("code", "name", "block", "capacity", "is_active")
    list_filter = ("block", "is_active")
    search_fields = ("code", "name")


@admin.register(BreakTimeSlot)
class BreakTimeSlotAdmin(admin.ModelAdmin):
    list_display = ("name", "start_time", "end_time", "is_active")
    list_filter = ("is_active",)


@admin.register(FoodItem)
class FoodItemAdmin(admin.ModelAdmin):
    list_display = ("name", "category", "price", "is_available", "prep_time_mins")
    list_filter = ("category", "is_available")
    search_fields = ("name",)


class FoodOrderItemInline(admin.TabularInline):
    model = FoodOrderItem
    extra = 0


@admin.register(FoodOrder)
class FoodOrderAdmin(admin.ModelAdmin):
    list_display = ("id", "student", "slot", "order_mode", "delivery_location", "status", "total_amount", "ordered_at")
    list_filter = ("status", "slot", "order_mode")
    inlines = [FoodOrderItemInline]


@admin.register(FoodOrderItem)
class FoodOrderItemAdmin(admin.ModelAdmin):
    list_display = ("order", "food_item", "quantity", "unit_price")


@admin.register(Payment)
class PaymentAdmin(admin.ModelAdmin):
    list_display = ("id", "order", "method", "status", "amount", "transaction_id", "paid_at")
    list_filter = ("status", "method")
    search_fields = ("transaction_id",)


@admin.register(MakeupClassSession)
class MakeupClassSessionAdmin(admin.ModelAdmin):
    list_display = ("date", "start_time", "end_time", "faculty", "class_section", "remedial_code")
    list_filter = ("date", "faculty")
    search_fields = ("remedial_code",)


@admin.register(MakeupAttendance)
class MakeupAttendanceAdmin(admin.ModelAdmin):
    list_display = ("session", "student", "mode", "marked_at")
    list_filter = ("mode",)
