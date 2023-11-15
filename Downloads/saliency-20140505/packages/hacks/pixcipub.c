/*                                                            
 *                                                            
 *      pixcipub.c     External        16-Feb-2006            
 *                                                            
 *      Copyright (C)  2006  EPIX, Inc.  All rights reserved. 
 *                                                            
 *      Frame Grabber Driver: Linux Device Driver wrappers.   
 *                                                            
 *      THIS IS A MACHINE GENERATED COPY                      
 */                                                           
#define PIXCI
#include <linux/version.h>
#include <linux/module.h>
#include <linux/param.h>
#include <linux/sched.h>
#include <linux/kernel.h>
#include <linux/pci.h>
#include <linux/ioport.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/delay.h>
#include <asm/uaccess.h>
#include <asm/current.h>
#if (LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,0))
  #include <asm/irq.h>
  #include <linux/moduleparam.h>
  #include <linux/interrupt.h>
  #if !defined(UTS_RELEASE)
    #include <generated/utsrelease.h>
    #endif
  #if !defined(SA_SHIRQ)
    #define SA_SHIRQ IRQF_SHARED
  #endif
#else
  EXPORT_NO_SYMBOLS;
#endif
#if defined(PIXCI)
  #include "pixcipub.h"
  char *PIXCIPARM = "";
  #if (LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,0))
    module_param(PIXCIPARM, charp, 0);
  #else
    MODULE_PARM(PIXCIPARM, "s");
  #endif
  MODULE_AUTHOR("EPIX, Inc.");
  MODULE_DESCRIPTION("PIXCI(R) 32 Bit Driver" ". " "3.0.00" " " "[08.07.11]" ". " "Copyright\251 2007 EPIX, Inc.");
  MODULE_SUPPORTED_DEVICE("PIXCI(R) Imaging Boards");
  #if defined(MODULE_LICENSE)
    MODULE_LICENSE("Proprietary");
  #endif
#endif
int init_pxdrvlnx(void)
{
    extern struct file_operations pxdrv_fops;
    int r;
    if (1)
 printk("init: " "PIXCI(R) 64 Bit Driver" " V" "3.0.00" " " UTS_RELEASE " " "Jul 11 2008" " " "14:09:23" "\n");
    r = register_chrdev(0, "PIXCI(R)", &pxdrv_fops);
    if (r < 0)
 return(r);
    #if (LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,0))
 return(wrapper_init_pxdrvlnx(r, PIXCIPARM, HZ, PAGE_SIZE, IRQ_HANDLED, IRQ_NONE));
    #else
 return(wrapper_init_pxdrvlnx(r, PIXCIPARM, HZ, PAGE_SIZE, 0, 0));
    #endif
}
void cleanup_pxdrvlnx(void)
{
    int major = wrapper_cleanup_pxdrvlnx();
    if (major != 0)
 unregister_chrdev(major, "PIXCI(R)");
    if (1)
 printk("cleanup: " "PIXCI(R) 64 Bit Driver" " V" "3.0.00" "\n");
}
module_init(init_pxdrvlnx);
module_exit(cleanup_pxdrvlnx);
long pxdrv_ioctl(struct file *filep, uint cmd, ulong arg)
{
    return(wrapper_pxdrv_ioctl(cmd, arg));
}
int pxdrv_open(struct inode *ip, struct file *filep)
{
    return(wrapper_pxdrv_open());
}
int pxdrv_release(struct inode *ip, struct file *filep)
{
    return(wrapper_pxdrv_release());
}
#if (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,0))
void pxdrv_vma_open(struct vm_area_struct *vma)
{
    MOD_INC_USE_COUNT;
}
void pxdrv_vma_close(struct vm_area_struct *vma)
{
    MOD_DEC_USE_COUNT;
}
#endif

#if (LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,0)) && (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,26))
struct page * pxdrv_vma_nopage(struct vm_area_struct *vma, ulong address, int *type)
{
  struct page *page;
  void *virt;
  virt = wrapper_pxdrv_mmapdope1(vma->vm_pgoff<<PAGE_SHIFT);
  if (virt == NULL) return(NULL);
  page = virt_to_page(virt + address - vma->vm_start);
  get_page(page);
  if (type) *type = VM_FAULT_MINOR;
  return(page);
}
struct vm_operations_struct pxdrv_vmops = {
 nopage: pxdrv_vma_nopage,
};
#endif

#if (LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,26))
int pxdrv_vma_fault(struct vm_area_struct *vma, struct vm_fault *vmf)
{
  void *virt;
  virt = wrapper_pxdrv_mmapdope1(vma->vm_pgoff<<PAGE_SHIFT);
  if (virt == NULL) return(VM_FAULT_SIGBUS);
  vmf->page = virt_to_page(virt + (ulong)vmf->virtual_address - vma->vm_start);
  get_page(vmf->page);
  return(0);
}
struct vm_operations_struct pxdrv_vmops = {
 fault: pxdrv_vma_fault,
};
#endif



int pxdrv_mmap(struct file *filep, struct vm_area_struct *vma)
{
    #if (LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,0))
 if (wrapper_pxdrv_mmapdope0(vma->vm_start, vma->vm_end, vma->vm_pgoff) == 'e') {
     vma->vm_flags |= VM_RESERVED;
     if (!remap_pfn_range(vma, vma->vm_start, vma->vm_pgoff, vma->vm_end-vma->vm_start, vma->vm_page_prot))
  return(0);
     return(-EAGAIN);
 } else {
     #if defined(__GFP_COMP)
  if (__GFP_COMP != 0) {
      vma->vm_flags |= VM_RESERVED;
      vma->vm_ops = &pxdrv_vmops;
      return(0);
  }
     #endif
 }
    #endif
    return(-EAGAIN);
}
struct file_operations pxdrv_fops = {
    unlocked_ioctl: pxdrv_ioctl,
    open: pxdrv_open,
    release: pxdrv_release,
    mmap: pxdrv_mmap,
    owner: THIS_MODULE,
};
int _cfunregparm0 wrapper_request_mem_region(ulong start, ulong len, char *name)
{
    return(request_mem_region(start, len, name) != NULL);
}
void _cfunregparm0 wrapper_release_mem_region(ulong start, ulong len)
{
    release_mem_region(start, len);
}
int _cfunregparm0 wrapper_request_region(ulong start, ulong len, char *name)
{
    return(request_region(start, len, name) != NULL);
}
void _cfunregparm0 wrapper_release_region(ulong start, ulong len)
{
    release_region(start, len);
}
void* _cfunregparm0 wrapper_ioremap(ulong adrs, ulong size)
{
    return(ioremap(adrs, size));
}
void _cfunregparm0 wrapper_iounmap(void *adrs)
{
    iounmap(adrs);
}
int _cfunregparm0 wrapper_copy_to_user(void *to, const void *from, ulong count)
{
    return(__copy_to_user(to, from, count));
}
int _cfunregparm0 wrapper_copy_from_user(void *to, const void *from, ulong count)
{
    return(__copy_from_user(to, from, count));
}
int _cfunregparm0 wrapper_pci_write_config_dword(struct pci_dev *dev, int where, u32 value)
{
    return(pci_write_config_dword(dev, where, value));
}
int _cfunregparm0 wrapper_pci_read_config_dword(struct pci_dev *dev, int where, u32 *value)
{
    return(pci_read_config_dword(dev, where, value));
}
int _cfunregparm0 wrapper_pci_enable_device(struct pci_dev *dev)
{
    return(pci_enable_device(dev));
}
struct pci_dev *
_cfunregparm0 wrapper_pci_find_device(uint vendor, uint device, struct pci_dev *last)
{
    return(pci_get_device(vendor, device, (struct pci_dev*)last));
}
int _cfunregparm0 _cfunregparm0 wrapper_pci_present(void)
{
    #if (LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,0))
 return(1);
    #else
 return(pci_present());
    #endif
}
void _cfunregparm0 _cfunregparm0 wrapper_down(struct semaphore *sem)
{
    down(sem);
}
void _cfunregparm0 wrapper_up(struct semaphore *sem)
{
    up(sem);
}
void _cfunregparm0 wrapper_add_timer(struct timer_list *timer, void (*func)(ulong data), ulong data, ulong millis)
{
    timer->data = data;
    timer->function = func;
    timer->expires = ((HZ*(millis+500)/1000)+500)/1000;
    timer->expires = timer->expires > 1? timer->expires: 1;
    timer->expires += jiffies;
    add_timer(timer);
}
void _cfunregparm0 wrapper_init_timer(struct timer_list *timer)
{
    init_timer(timer);
}
void _cfunregparm0 wrapper_del_timer_sync(struct timer_list *timer)
{
    del_timer_sync(timer);
}
int _cfunregparm0 wrapper_send_sig(int sig, struct task_struct *task, int z)
{
    return(send_sig(sig, task, z));
}
struct task_struct *
_cfunregparm0 wrapper_get_current(void)
{
    return(get_current());
}
#if (LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,0))
typedef irqreturn_t (*irqhandler_t)(int, void*, struct pt_regs *);
#else
typedef void (*irqhandler_t)(int, void*, struct pt_regs *);
#endif
int _cfunregparm0 wrapper_request_irq(uint irq, irq_handler_t handler,int shareirq, const char* dev_name, void *dev_id) {
    return(request_irq(irq, handler, shareirq?SA_SHIRQ:0, dev_name, dev_id));
}
void _cfunregparm0 wrapper_free_irq(uint irq, void *dev_id)
{
    free_irq(irq, dev_id);
}
int _cfunregparm0 wrapper_put_user_long(long value, long *ptr)
{
    return(__put_user(value, ptr));
}
ulong _cfunregparm0 wrapper_get_free_pages(int nowarn, uint order, int dma32)
{
    #if defined(__GFP_NOWARN)
 #define WRAPPER_GFP_NOWARN __GFP_NOWARN
    #else
 #define WRAPPER_GFP_NOWARN 0
    #endif
    #if defined(__GFP_COMP)
 #define WRAPPER_GFP_COMP __GFP_COMP
    #else
 #define WRAPPER_GFP_COMP 0
    #endif
    #if defined(__GFP_DMA32)
 #define WRAPPER_GFP_DMA32 __GFP_DMA32
    #else
 #define WRAPPER_GFP_DMA32 0
    #endif
    ulong adrs = __get_free_pages( GFP_KERNEL
      | (nowarn? WRAPPER_GFP_NOWARN: 0)
      | (dma32? WRAPPER_GFP_DMA32: 0)
      | WRAPPER_GFP_COMP, order);
    return(adrs);
    #undef WRAPPER_GFP_NOWARN
    #undef WRAPPER_GFP_COMP
}
void _cfunregparm0 wrapper_free_pages(ulong adrs, uint order)
{
    free_pages(adrs, order);
}
ulong _cfunregparm0 wrapper_virt_to_bus(void *p)
{
    return(virt_to_bus(p));
}
struct semaphore *
 _cfunregparm0 wrapper_kmalloc_semaphore(int value)
{
    struct semaphore *sem = kmalloc(sizeof(struct semaphore), GFP_KERNEL);
    if (sem)
 sema_init(sem, value);
    return(sem);
}
struct timer_list *
 _cfunregparm0 wrapper_kmalloc_timer_list(void)
{
    return((struct timer_list *)kmalloc(sizeof(struct timer_list), GFP_KERNEL));
}
int _cfunregparm0 wrapper_pci_resource_flags_io(struct pci_dev *dev, int index)
{
    return(pci_resource_flags(dev, index)&IORESOURCE_IO);
}
ulong _cfunregparm0 wrapper_pci_resource_start(struct pci_dev *dev, int index)
{
    return(pci_resource_start(dev, index));
}
ulong _cfunregparm0 wrapper_pci_resource_len(struct pci_dev *dev, int index)
{
    return(pci_resource_len(dev, index));
}
int _cfunregparm0 wrapper_pci_dev_irq(struct pci_dev *dev)
{
    return(dev->irq);
}
int _cfunregparm0 wrapper_pci_dev_bus_number(struct pci_dev *dev)
{
    if (dev->bus)
 return(dev->bus->number);
    return(0);
}
ulong _cfunregparm0 wrapper_jiffies(void) { return(jiffies); }
ulong _cfunregparm0 wrapper_HZ(void) { return(HZ); }
char * _cfunregparm0 wrapper_UTS_RELEASE(void) { return(UTS_RELEASE); }
ulong _cfunregparm0 wrapper_num_physpages(void) { return(num_physpages); }
u32 _cfunregparm0 wrapper_inl(ulong port) { return(inl(port)); }
void _cfunregparm0 wrapper_outl(u32 data, ulong port) { outl(data, port); }
u32 _cfunregparm0 wrapper_readl(void *adrs) { return(readl(adrs));}
void _cfunregparm0 wrapper_writel(u32 data, void *adrs) { writel(data, adrs); }
void * _cfunregparm0 wrapper_kmalloc(size_t size) { return(kmalloc(size, GFP_KERNEL)); }
void _cfunregparm0 wrapper_kfree(void *ptr) { kfree(ptr); }
void _cfunregparm0 wrapper_disable_irq(int irq) { disable_irq(irq); }
void _cfunregparm0 wrapper_enable_irq(int irq) { enable_irq(irq); }
void _cfunregparm0 wrapper_schedule(void) { schedule(); }
void _cfunregparm0 wrapper_gettimeofday(time_t *ssecp, long *usecp)
{
    struct timeval t;
    do_gettimeofday(&t);
    *ssecp = t.tv_sec;
    *usecp = t.tv_usec;
}
void _cfunregparm0 wrapper_mdelay(ulong msecs) { mdelay(msecs); }
void _cfunregparm0 wrapper_udelay(ulong usecs) { udelay(usecs); }
void _cfunregparm0 wrapper_msleep(uint msecs)
{
    #if (LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,0))
 msleep(msecs);
    #else
 mdelay(msecs);
    #endif
}
